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
from tasks.vqa import VQA, get_data_tuple, DataTuple
from torch.optim import Adam


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

fewshot_qs_dir = "./data/vqa/fewshot"

def get_data_tuple_lists(splits: str, bs:int, shuffle=False, drop_last=False, frcnn_cfg=False, imgfeat_dir:str=None) -> DataTuple:
    tuples = {
        'train': [],
        'eval': []
    }
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
                img_feat_fn = os.path.join(imgfeat_dir, splits, pokemon_name)
                image_features = load_obj_npy(img_feat_fn)
                tset = VQATorchDataset(dset, imgfeat_dir=imgfeat_dir, image_features=image_features)
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
            self.train_train_tuples, self.train_eval_tuples = get_data_tuple_lists(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True,
                frcnn_cfg=self.frcnn_cfg, imgfeat_dir=args.image_features,
            )
            if args.valid != "":
                self.valid_train_tuples, self.valid_eval_tuples = get_data_tuple_lists(
                    args.valid, bs=1024,
                    shuffle=False, drop_last=False,
                    frcnn_cfg=self.frcnn_cfg,  imgfeat_dir=args.image_features,
                )
            else:
                self.valid_train_tuples, self.valid_eval_tuples = None, None
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
            # if 'bert' in args.optim:
            #     batch_per_epoch = len(self.valid_train_tuples[0].loader)
            #     t_total = int(batch_per_epoch * args.epochs)
            #     print("BertAdam Total Iters: %d" % t_total)
            #     from lxrt.optimization import BertAdam
            #     self.optim = BertAdam(list(p for p in self.model.parameters() if p.requires_grad),
            #                         lr=args.lr,
            #                         warmup=0.1,
            #                         t_total=t_total)
            # else:
            self.optim = Adam([p for p in self.model.parameters() if p.requires_grad], args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
    
    def fewshot_evaluate(self, train_tuples: List[DataTuple], eval_tuples: List[DataTuple], num_updates: int = 1):
        """
        Given paired train/eval data, trains model for `num_updates` updates on train data, then evaluates on eval data.
        Returns evaluation accuracy.
        """
        train_scores = []
        eval_scores = []
        # for epoch in range(args.meta_epochs):
        ori_model = copy.deepcopy(self.model)
        for (train_tuple, eval_tuple) in zip(train_tuples, eval_tuples):
            self.train(train_tuple, train_tuple)

            eval_score = self.evaluate(eval_tuple)
            print(f"Evaluation Score: {eval_score}")
            eval_scores.append(eval_score)
            self.model = ori_model
        print(f"Average Evaluation Score: {sum(eval_scores) / len(eval_scores)}")



if __name__ == "__main__":
    # Build Class

    vqa = MetaVQA(args.interact, args.test)

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
        # if 'test' in args.test:
        #     vqa.predict(
        #         get_data_tuple(args.test, bs=950,
        #                        shuffle=False, drop_last=False),
        #         dump=os.path.join(args.output, 'test_predict.json')
        #     )
        # elif 'val' in args.test:
        fewshot_val_train, fewshot_val_test = get_data_tuple_lists(args.test, bs=950, shuffle=False, drop_last=False, imgfeat_dir=args.image_features)
        result = vqa.fewshot_evaluate(
            fewshot_val_train,
            fewshot_val_test,
            num_updates = args.num_fewshot_updates
        )
        print(result)
    else:
        if vqa.valid_train_tuples is not None:
            print("Valid Oracle: %0.2f" % (vqa.fewshot_oracle_score(vqa.valid_train_tuples, vqa.valid_eval_tuples) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.fewshot_train(vqa.train_train_tuples, vqa.train_eval_tuples, vqa.valid_train_tuples, vqa.valid_eval_tples)


