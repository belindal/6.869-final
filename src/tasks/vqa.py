# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQARawTorchDataset, VQAEvaluator, MSCOCO_IMGFEAT_ROOT, SPLIT2NAME
from utils import load_obj_tsv, load_obj_npy
from glob import glob
import numpy as np
import json
from frcnn.extract_features_frcnn import FeatureExtractor
from frcnn.frcnn_utils import Config
from frcnn.modeling_frcnn import GeneralizedRCNN
from frcnn.processing_image import Preprocess


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, frcnn_cfg=False) -> DataTuple:
    dset = VQADataset(splits)
    if frcnn_cfg: 
        tset = VQARawTorchDataset(dset, frcnn_cfg=frcnn_cfg)
    else:
        tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self, interact):
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
        if not interact:
            self.train_tuple = get_data_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True,
                frcnn_cfg=self.frcnn_cfg,
            )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False,
                frcnn_cfg=self.frcnn_cfg,
            )
        else:
            self.valid_tuple = None
        label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        num_answers = len(label2ans)
        
        # Model
        self.model = VQAModel(num_answers, frcnn=frcnn, frcnn_cfg=self.frcnn_cfg)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if not interact:
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                self.optim = BertAdam(list(p for p in self.model.parameters() if p.requires_grad),
                                    lr=args.lr,
                                    warmup=0.1,
                                    t_total=t_total)
            else:
                self.optim = args.optimizer([p for p in self.model.parameters() if p.requires_grad], args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple=None):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, datum_tuple in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()

                if args.load_frcnn:
                    (ques_id, images, sizes, scales_yx, sent, target) = datum_tuple
                    model_inputs = {'images': images.cuda(), 'sizes': sizes.cuda(), 'scales_yx': scales_yx.cuda(), 'sent': sent}
                    self.model.frcnn.eval()
                else:
                    (ques_id, feats, boxes, sent, target) = datum_tuple
                    model_inputs = {'feat': feats.cuda(), 'pos': boxes.cuda(), 'sent': sent}
                target = target.cuda()

                logit, frcnn_features = self.model(**model_inputs)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            train_score = evaluator.evaluate(quesid2ans) * 100.
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if eval_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

                print(log_str, end='')

                with open(self.output + "/log.log", 'a') as f:
                    f.write(log_str)
                    f.flush()
                if abs(valid_score - 1) < 1e-4:
                    break
            else:
                print(log_str, end='')
                if abs(train_score - 100) < 1e-4:
                    self.save("BEST")
                    break

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            if args.load_frcnn:
                (ques_id, images, sizes, scales_yx, sent) = datum_tuple[:5]
                model_inputs = {'images': images.cuda(), 'sizes': sizes.cuda(), 'scales_yx': scales_yx.cuda(), 'sent': sent}
            else:
                (ques_id, feats, boxes, sent) = datum_tuple[:4]
                model_inputs = {'feat': feats.cuda(), 'pos': boxes.cuda(), 'sent': sent}
            with torch.no_grad():
                logit, frcnn_features = self.model(**model_inputs)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
            evaluator.dump_result(quesid2ans, dump[:-5]+"_human_readable.json", human_readable=True)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)
    
    def interact(self, imgid2img, ans2label, label2ans, adversarial=False, train=True):
        self.model.eval()
        n_iter = 10
        step_size = 1e-5  # TODO tune
        k=10
        if adversarial:
            for param in self.model.parameters(): param.requires_grad = True
        while True:
            imgid = input("Image ID: ")
            if imgid not in imgid2img:
                imgid = f"COCO_val2014_000000{imgid}"
                if imgid not in imgid2img:
                    print("Image not found! Try again!")
                    continue
            
            if args.load_frcnn:
                #TODO
                img_info = imgid2img[imgid]
                model_inputs = {k: img_info[k].clone().unsqueeze(0).cuda() for k in ['images', 'sizes', 'scales_yx']}
            else:
                # Get image info
                img_info = imgid2img[imgid]
                obj_num = img_info['num_boxes']
                feats = img_info['features'].copy()
                boxes = img_info['boxes'].copy()
                assert obj_num == len(boxes) == len(feats)

                # Normalize the boxes (to 0 ~ 1)
                img_h, img_w = img_info['img_h'], img_info['img_w']
                boxes = boxes.copy()
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                np.testing.assert_array_less(boxes, 1+1e-5)
                np.testing.assert_array_less(-boxes, 0+1e-5)
                feats, boxes = torch.tensor(feats).cuda(), torch.tensor(boxes).cuda()
                model_inputs = {'feat': feats.unsqueeze(0), 'pos': boxes.unsqueeze(0)}
            sent = ""
            while sent != "new":
                sent = input("Question: ")
                with torch.no_grad():
                    model_inputs['sent'] = [sent]
                    logit, frcnn_features = self.model(**model_inputs)
                    score, label = logit.max(1)
                    topk_score, topk_label = logit.squeeze(0).topk(k)
                    ans = label2ans[label]
                print(f"Answer: {ans}")
                topk_labels_print = '\n\t'.join([f'{label2ans[label]}: {topk_score[idx]}' for idx, label in enumerate(topk_label)])
                print(f"Top-K prediction:\n\t{topk_labels_print}")
                if adversarial:
                    desired_target = None
                    while not desired_target in ans2label:
                        desired_target = input(f"Desired Target: ")
                    desired_label = torch.zeros(logit.size()).cuda()
                    desired_label[0,ans2label[desired_target]] = 1
                    logit, frcnn_features = self.model(**model_inputs)
                    import pdb; pdb.set_trace()
                    for i in tqdm(range(n_iter)):
                        loss = self.bce_loss(logit, desired_label)
                        gradient = torch.autograd.grad(loss, model_inputs['images'])[0]
                        model_inputs['images'] -= step_size * gradient
                        logit, frcnn_features = self.model(**model_inputs)
                        topk_score, topk_label = logit.squeeze(0).topk(k, dim=1)
                        topk_labels_print = '\n\t'.join([f'{label2ans[label]}: {topk_score[idx]}' for idx, label in enumerate(topk_label)])
                        print(f"Step {i}/{n_iter} Top-K prediction:\n\t{topk_labels_print}")
                    import pdb; pdb.set_trace()
                    np.save(model_inputs['images'], os.path.join('adversarial_outputs', f'{imgid}.jpg'))
                

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            if args.load_frcnn:
                (ques_id, images, sizes, scales_yx, sent, target) = datum_tuple
            else:
                (ques_id, feats, boxes, sent, target) = datum_tuple
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class

    vqa = VQA(args.interact)

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

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
    elif args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


