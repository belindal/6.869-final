# coding=utf-8
# Copyleft 2019 project LXRT.
from typing import List

import os
import collections
import copy

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import learn2learn as l2l

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQARawTorchDataset, VQAEvaluator, MSCOCO_IMGFEAT_ROOT, SPLIT2NAME, MetaVQADataset
from src.lxrt.entry import convert_sents_to_features
from utils import load_obj_tsv, load_obj_npy
from glob import glob
import numpy as np
import json
from frcnn.extract_features_frcnn import FeatureExtractor
from frcnn.frcnn_utils import Config
from frcnn.modeling_frcnn import GeneralizedRCNN
from frcnn.processing_image import Preprocess
from tasks.vqa import VQA, get_data_tuple, DataTuple
from torch.optim import Adam


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

fewshot_qs_dir = "./data/vqa/fewshot"
img_raw_dir = "PokemonData"
all_pokemon_list = []
for split in ["train", "val", "test"]:
    all_pokemon_list.extend([os.path.split(fp)[-1] for fp in glob(os.path.join(img_raw_dir, split, "*"))])

def get_data_tuple_lists(splits: str, bs:int, shuffle=False, drop_last=False, frcnn_cfg=False, imgfeat_dir:str=None) -> DataTuple:
    tuples = {
        'train': [],
        'eval': []
    }
    all_image_features = []
    print(f"Loading images from {os.path.join(imgfeat_dir, splits)}")
    for pokemon in glob(os.path.join(imgfeat_dir, splits, "*")):
        # img_feat_fn = os.path.join(imgfeat_dir, splits, pokemon_name)
        image_features = load_obj_npy(pokemon)
        all_image_features.extend(image_features)
    print(f"Loading questions from {fewshot_qs_dir}/{splits}")
    for data_fn in glob(f'{fewshot_qs_dir}/{splits}/*.json'):
        loaded_data = json.load(open(data_fn))
        pokemon_name = os.path.split(data_fn)[-1].replace('.json', '')
        for fs_split in tuples:
            dset = VQADataset(data_fn, data=loaded_data[fs_split])
            if frcnn_cfg: 
                # TODO
                import pdb; pdb.set_trace()
                tset = VQARawTorchDataset(dset, frcnn_cfg=frcnn_cfg)
            else:
                assert imgfeat_dir is not None
                image_features = []
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 #if (split == 'minival' and topk is None) else topk
                tset = VQATorchDataset(dset, imgfeat_dir=imgfeat_dir, image_features=all_image_features)
            evaluator = VQAEvaluator(dset)
            data_loader = DataLoader(
                tset, batch_size=bs,
                shuffle=shuffle, num_workers=args.num_workers,
                drop_last=drop_last, pin_memory=True
            )
            train_tuple = DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)
            tuples[fs_split].append(train_tuple)
    return tuples['train'], tuples['eval']


class MetaVQA(VQA):
    def __init__(self, interact, test):
        # Load FRCNN weights
        self.frcnn_cfg = None
        frcnn = None
        if args.load_frcnn:
            self.frcnn_cfg = Config.from_pretrained(
                FeatureExtractor.CONFIG_URL.get("FRCNN", "FRCNN")
            )
            self.frcnn_cfg.model.device = 'cuda'
            frcnn = GeneralizedRCNN.from_pretrained(
                FeatureExtractor.MODEL_URL.get("FRCNN", "FRCNN"),
                config=self.frcnn_cfg,
            )

        # Datasets
        if not interact and not test:
            self.train_support_tuples, self.train_query_tuples = get_data_tuple_lists(
                args.train, bs=args.batch_size, shuffle=True, drop_last=False,
                frcnn_cfg=self.frcnn_cfg, imgfeat_dir=args.image_features,
            )
            self.train_data = DataLoader(
                MetaVQADataset(self.train_support_tuples, self.train_query_tuples), batch_size=1,
                shuffle=False, num_workers=args.num_workers,
                drop_last=False, pin_memory=True
            )
            if args.valid != "":
                self.valid_support_tuples, self.valid_query_tuples = get_data_tuple_lists(
                    args.valid, bs=1024,
                    shuffle=False, drop_last=False,
                    frcnn_cfg=self.frcnn_cfg,  imgfeat_dir=args.image_features,
                )
                self.valid_data = DataLoader(
                    MetaVQADataset(self.valid_support_tuples, self.valid_query_tuples), batch_size=1,
                    shuffle=False, num_workers=args.num_workers,
                    drop_last=False, pin_memory=True
                )
            else:
                self.valid_support_tuples, self.valid_query_tuples = None, None
                self.valid_data = None
        label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        num_answers = len(label2ans)
        
        # Model
        self.model = VQAModel(num_answers, frcnn=frcnn, frcnn_cfg=self.frcnn_cfg)
        if args.learn_word_embeds_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.lxrt_encoder.model.bert.embeddings.word_embeddings.parameters():
                p.requires_grad = True

        self.maml = None
        if not args.interact and not args.test:
            if args.meta_word_embeds_only or args.learn_word_embeds_only:
                self.maml = l2l.algorithms.MAML(self.model.lxrt_encoder.model.bert.embeddings.word_embeddings, lr=(args.meta_lr))
            else:
                self.maml = l2l.algorithms.MAML(self.model, lr=(args.meta_lr))

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if self.maml: self.maml = self.maml.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if not interact:
            # if 'bert' in args.optim:
            #     batch_per_epoch = len(self.valid_support_tuples[0].loader)
            #     t_total = int(batch_per_epoch * args.epochs)
            #     print("BertAdam Total Iters: %d" % t_total)
            #     from lxrt.optimization import BertAdam
            #     self.optim = BertAdam(list(p for p in self.model.parameters() if p.requires_grad),
            #                         lr=args.lr,
            #                         warmup=0.1,
            #                         t_total=t_total)
            # else:
            # for n,p in self.model.named_parameters():
            #     if n != "lxrt_encoder.model.bert.embeddings.word_embeddings.weight":
            #         p.requires_grad = False
            base_trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if self.maml:
                meta_trainable_params = [p for p in self.maml.parameters() if p.requires_grad]
                # assert len(base_trainable_params) == 1
                self.meta_optim = Adam(meta_trainable_params, args.meta_lr)
            self.optim = Adam(base_trainable_params, args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
    
    def compute_loss(
        self, model, loader, use_tqdm=False
    ):
        # dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if use_tqdm else (lambda x: x)

        losses = 0
        for i, datum_tuple in iter_wrapper(enumerate(loader)):
            if args.load_frcnn:
                (ques_id, images, sizes, scales_yx, sent, target) = datum_tuple
                model_inputs = {'images': images.cuda(), 'sizes': sizes.cuda(), 'scales_yx': scales_yx.cuda(), 'sent': sent}
                model.frcnn.eval()
            else:
                (ques_id, feats, boxes, sent, target) = datum_tuple
                model_inputs = {'feat': feats.cuda(), 'pos': boxes.cuda(), 'sent': sent}
            target = target.cuda()

            if args.meta_word_embeds_only or args.learn_word_embeds_only:
                train_features = convert_sents_to_features(sent, self.model.lxrt_encoder.max_seq_length, self.model.lxrt_encoder.tokenizer)
                input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
                word_embeds = model(input_ids)
                model_inputs['precomputed_word_embeddings'] = word_embeds
                logit, frcnn_features = self.model(**model_inputs)
            else:
                logit, frcnn_features = model(**model_inputs)
            assert logit.dim() == target.dim() == 2
            loss = self.bce_loss(logit, target)
            loss = loss * logit.size(1)
            losses += loss
        return losses
    

    def fewshot_train(
        self, train_support_tuples: List[DataTuple], train_query_tuples: List[DataTuple],
        valid_support_tuples: List[DataTuple] = None, valid_query_tuples: List[DataTuple] = None, num_fs_updates: int = 1,
        init_eval_score: float = 0.0,
    ):
        # train_scores = []
        train_losses = []
        valid_scores = []
        valid_losses = []
        best_valid_score = init_eval_score
        for epoch in range(args.meta_epochs):
            self.optim = Adam([p for p in self.model.parameters() if p.requires_grad], args.lr)
            print(f"=== EPOCH {epoch} ===")
            # train_scores.append(0)
            train_loss = 0
            old_model = copy.deepcopy(self.model)
            tqdm_bar = tqdm(zip(train_support_tuples, train_query_tuples))

            # TODO batch up tasks???
            # for (train_support_batch, train_query_batch) in collate_support_query(train_support_tuples, train_query_tuples):
            for (train_support_tuple, train_query_tuple) in tqdm_bar:
                # import pdb; pdb.set_trace()
                # self.train(train_support_tuple, epochs=1, do_save=False, do_break=True)
                # loss = self.compute_loss(self.model, train_query_tuple[1]); loss.backward(); self.optim.step()
                # for k in self.model.state_dict(): assert (self.model.state_dict()[k] == old_model.state_dict()[k]).all()
                task_model = self.maml.clone()  # torch.clone() for nn.Modules

                self.model.train()
                self.meta_optim.zero_grad()
                _, train_support_loader, _ = train_support_tuple
                for step in range(5):
                    adaptation_loss = self.compute_loss(task_model, train_support_loader, use_tqdm=False)
                    task_model.adapt(adaptation_loss, allow_unused=True)  # computes gradient, update task_model in-place
                # Sum (over tasks)?
                _, train_query_loader, _ = train_query_tuple
                query_loss = self.compute_loss(task_model, train_query_loader, use_tqdm=False)
                query_loss.backward()  # gradients w.r.t. maml.parameters()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.meta_optim.step()
                # query_loss2 = self.compute_loss(self.model, train_query_loader)
                # query_loss2.backward()
                # self.optim.step()
                # train_loss.append(query_loss)
                train_loss += query_loss.detach().cpu().item()
                tqdm_bar.set_description("Loss : {:.3f} ".format(query_loss.detach().cpu().item()))
            train_loss /= len(train_support_tuples)
            # self.meta_optim.zero_grad()
            # train_loss.backward()
            # self.meta_optim.step()
            # train_losses.append(train_loss.detach().cpu().item())
            train_losses.append(train_loss)
            print(f"(Approximate) train loss: {train_loss}")
            ori_model = copy.deepcopy(self.model)
            
            self.save(f"{epoch}")
            if valid_support_tuples:
                valid_score_trials = []
                for trials in range(5):
                    valid_score_trials.append(self.fewshot_evaluate(valid_support_tuples, valid_query_tuples, num_fs_updates=num_fs_updates))
                valid_score = sum(valid_score_trials) / len(valid_score_trials)
                print(f"Avg. valid score: {valid_score}")
                valid_scores.append(valid_score)
                if valid_score > best_valid_score:
                    print("NEW BEST MODEL")
                    self.model = ori_model
                    self.save("BEST")
                    best_valid_score = valid_score
        self.save("LAST")

        """
        # task_num, setsz, c_, h, w = train_support_tuples.size()
        querysz = train_query_tuples.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for (train_support_tuple, train_query_tuple) in zip(train_support_tuples, train_query_tuples):
            import pdb; pdb.set_trace()
            dset, loader, evaluator = train_support_tuple

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(train_support_tuples[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, train_support_tuples[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(train_query_tuples[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, train_query_tuples[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, train_query_tuples[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(train_query_tuples[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, train_query_tuples[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, train_query_tuples[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1,self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(train_support_tuples[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, train_support_tuples[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(train_query_tuples[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, train_query_tuples[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, train_query_tuples[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / len(train_support_tuples)

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * len(train_support_tuples))

        return accs
        """
    
    def fewshot_evaluate(self, support_tuples: List[DataTuple], query_tuples: List[DataTuple], num_fs_updates: int = 1, dump_dir: str = None):
        """
        Given paired support/query data, trains model for `num_fs_updates` updates on train data, then evaluates on eval data.
        Returns evaluation accuracy.
        """
        sup_scores = []
        qu_scores = []
        do_break = False
        for (sup_tuple, query_tuple) in tqdm(zip(support_tuples, query_tuples)):
            self.optim = Adam([p for p in self.model.parameters() if p.requires_grad], args.lr)

            self.optim.zero_grad()
            ori_model = copy.deepcopy(self.model)
            if do_break: import pdb; pdb.set_trace()
            self.train(sup_tuple, sup_tuple, use_tqdm=False, epochs=num_fs_updates, do_save=False, do_break=do_break)
            # for k in self.model.state_dict(): assert (self.model.state_dict()[k] == ori_model.state_dict()[k]).all()
            if do_break: import pdb; pdb.set_trace()

            sup_score = self.evaluate(sup_tuple)
            # print(f"Train Score: {train_score}")
            sup_scores.append(sup_score)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                pokemon = query_tuple.dataset.data[0]['img_id'].split('/')[-2]
                assert query_tuple.dataset.data[0]['sent'] == f"Is this a {pokemon}?"
                assert query_tuple.dataset.data[0]['label'] == {"yes": 1}
                dump_file = os.path.join(dump_dir, f'{pokemon}.json')
            else:
                dump_file = None
            q_score = self.evaluate(query_tuple, dump=dump_file)
            # print(f"Evaluation Score: {eval_score}")
            qu_scores.append(q_score)
            self.model = ori_model
        avg_q_score = sum(qu_scores) / len(qu_scores)
        print(f"Average Support Score: {sum(sup_scores) / len(sup_scores)}")
        print(f"Average Query Score: {avg_q_score}")
        return avg_q_score

    def load(self, path, all_pokemon_list=None):
        state_dict = torch.load("%s.pth" % path)
        added_pokemon = False
        if all_pokemon_list and state_dict['lxrt_encoder.model.bert.embeddings.word_embeddings.weight'].size(0) > self.model.lxrt_encoder.model.bert.embeddings.word_embeddings.weight.size(0):
            print("Loading pokemon vocab")
            # Load new pokemon
            self.model.add_new_tokens(all_pokemon_list)
            added_pokemon = True
        self.model.load_state_dict(state_dict, strict=False)
        print("Load model from %s" % path)
        if all_pokemon_list and not added_pokemon:
            print("Loading pokemon vocab")
            self.model.add_new_tokens(all_pokemon_list)
        if args.learn_word_embeds_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.lxrt_encoder.model.bert.embeddings.word_embeddings.parameters():
                p.requires_grad = True
        if self.maml:
            if args.meta_word_embeds_only or args.learn_word_embeds_only:
                self.maml = l2l.algorithms.MAML(self.model.lxrt_encoder.model.bert.embeddings.word_embeddings, lr=(args.meta_lr))
            else:
                self.maml = l2l.algorithms.MAML(self.model, lr=(args.meta_lr))
            self.maml = self.maml.cuda()
            meta_trainable_params = [p for p in self.maml.parameters() if p.requires_grad]
            self.meta_optim = Adam(meta_trainable_params, args.meta_lr)
        base_trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # assert len(base_trainable_params) == 1
        self.optim = Adam(base_trainable_params, args.lr)


if __name__ == "__main__":
    # Build Class

    vqa = MetaVQA(args.interact, args.test)

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None and not args.epoch_sweep:
        print(args.add_pokemon_vocab)
        if not args.add_pokemon_vocab:
            all_pokemon_list = None
        vqa.load(args.load, all_pokemon_list)
        

    # Test or Train
    if args.interact:
        # Loading detection features to img_data
        img_data = []
        if args.load_frcnn:
            image_preprocess = Preprocess(vqa.frcnn_cfg)
            for fn in glob(f"img_dir/*.jpg")+glob(f"img_dir/*.png"):
                img_ids, images, sizes, scales_yx = image_preprocess(fn)
                assert len(img_ids) == len(images) == len(sizes) == len(scales_yx)
                for i in range(len(img_ids)):
                    img_datum = {
                        'img_id': img_ids[i],
                        'images': images[i],
                        'sizes': sizes[i],
                        'scales_yx': scales_yx[i],
                    }
                    img_data.append(img_datum)
        else:
            img_data.extend(load_obj_npy('frcnn_output'))
            # for split in ['minival']:
            #     tsv_file = os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split]))
            #     img_data.extend(load_obj_tsv(tsv_file, topk=5000))

        # Answers
        ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(ans2label) == len(label2ans)

        # Convert img list to dict
        imgid2img = {}
        for img_datum in img_data:
            imgid2img[img_datum['img_id']] = img_datum
        result = vqa.interact(imgid2img, ans2label, label2ans)
    elif args.test is not None or args.epoch_sweep:
        args.fast = args.tiny = False       # Always loading all data in test
        # if 'test' in args.test:
        #     vqa.predict(
        #         get_data_tuple(args.test, bs=950,
        #                        shuffle=False, drop_last=False),
        #         dump=os.path.join(args.output, 'test_predict.json')
        #     )
        # elif 'val' in args.test:
        if args.epoch_sweep:
            assert args.test is not None
            checkpoints = glob(os.path.join(args.epoch_sweep, "*.pth"))
            epochs = [
                int(os.path.split(checkpoint_fn)[-1].replace(".pth", "")) for checkpoint_fn in checkpoints
                if os.path.split(checkpoint_fn)[-1] != "BEST.pth" and os.path.split(checkpoint_fn)[-1] != "LAST.pth"
            ]
            max_epoch = max(epochs)
            loaded_checkpoint = False
        else:
            max_epoch = 1 
            loaded_checkpoint = True
        best_result = 0
        for epoch in range(max_epoch):
            if not loaded_checkpoint:
                if not args.add_pokemon_vocab: all_pokemon_list = None
                checkpoint_fn = os.path.join(args.epoch_sweep, str(epoch))
                vqa.load(checkpoint_fn, all_pokemon_list)
                print(epoch)
            fewshot_val_train, fewshot_val_test = get_data_tuple_lists(args.test, bs=950, shuffle=False, drop_last=False, imgfeat_dir=args.image_features)
            valid_score_trials = []
            for trials in range(5):
                if not loaded_checkpoint:
                    if not args.add_pokemon_vocab: all_pokemon_list = None
                    vqa.load(checkpoint_fn, all_pokemon_list)
                valid_score_trials.append(vqa.fewshot_evaluate(
                    fewshot_val_train,
                    fewshot_val_test,
                    num_fs_updates = args.num_fewshot_updates,
                    dump_dir = os.path.join(args.output, f'{args.test}_try{trials}_predict'),
                ))
            result = sum(valid_score_trials) / len(valid_score_trials)
            # vqa.fewshot_evaluate(fewshot_val_train, fewshot_val_test, num_fs_updates=args.num_fewshot_updates)
            print(result)
            if args.epoch_sweep and result > best_result:
                print("BEST EPOCH")
                if not args.add_pokemon_vocab: all_pokemon_list = None
                vqa.load(checkpoint_fn, all_pokemon_list)
                vqa.save("BEST")
                best_result = result
    else:
        init_eval_score = 0
        if vqa.valid_support_tuples is not None:
            valid_score_trials = []
            for trials in range(5):
                valid_score_trials.append(vqa.fewshot_evaluate(vqa.valid_support_tuples, vqa.valid_query_tuples, num_fs_updates=args.num_fewshot_updates))
                # valid_score_trials.append(vqa.fewshot_evaluate(vqa.train_support_tuples, vqa.train_query_tuples, num_fs_updates=args.num_fewshot_updates))
            init_eval_score = sum(valid_score_trials) / len(valid_score_trials)
            print("Valid Oracle: %0.2f" % (init_eval_score * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.fewshot_train(
            vqa.train_support_tuples, vqa.train_query_tuples, vqa.valid_support_tuples, vqa.valid_query_tuples, num_fs_updates=args.num_fewshot_updates, init_eval_score=init_eval_score
        )


