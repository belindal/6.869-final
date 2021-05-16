# coding=utf-8
# Copyleft 2019 project LXRT.
from typing import List

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
from frcnn.processing_image import Preprocess

from glob import glob
from tqdm import tqdm

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
IMG_DIR = 'img_dir/'
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'fewshot_train': 'fewshot_train',
    'train': 'train2014_4',
    'valid': 'val2014_4',
    'minival': 'val2014_4',
    'nominival': 'val2014_4',
    'test': 'test2015',
}


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str, data: list = None):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        if data:
            self.data = data
        else:
            self.data = []
            for split in self.splits:
                self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        # print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
Returns non-featurized images
"""
class VQARawTorchDataset(Dataset):
    def __init__(self, dataset: VQADataset, frcnn_cfg = None):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading raw images to img_data
        img_data = []
        image_preprocess = Preprocess(frcnn_cfg, device='cpu')
        # TODO change this loading...
        for split in dataset.splits:
            for fn in tqdm(glob(f"{os.path.join(IMG_DIR, SPLIT2NAME[split])}/*")):
                # for fn in tqdm(glob(f"{data_dir}/*.jpg")):
                # ckpt_save_dir = data_dir+"_tensorized"
                ckpt_save_dir = None
                if ckpt_save_dir and not os.path.exists(ckpt_save_dir):
                    os.makedirs(ckpt_save_dir, exist_ok=True)
                if not ckpt_save_dir or not os.path.exists(os.path.join(ckpt_save_dir, 'image_tensors.pt')):
                    img_ids, images, sizes, scales_yx = image_preprocess(fn, ckpt_save_dir=ckpt_save_dir)
                else:
                    img_ids, images, sizes, scales_yx = torch.load(os.path.join(ckpt_save_dir, 'image_tensors.pt'))
                # img_id = os.path.split(fn)[1].replace('.jpg', '')
                assert len(img_ids) == len(images) == len(sizes) == len(scales_yx)
                for i in range(len(img_ids)):
                    img_datum = {
                        'img_id': img_ids[i],
                        'images': images[i],
                        'sizes': sizes[i],
                        'scales_yx': scales_yx[i],
                    }
                    img_data.append(img_datum)
        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)

        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        img_info = self.imgid2img[img_id]
        image = img_info['images']
        size = img_info['sizes']
        scale_yx = img_info['scales_yx']

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, image, size, scale_yx, ques, target
        else:
            return ques_id, image, size, scale_yx, ques


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset, imgfeat_dir: str = MSCOCO_IMGFEAT_ROOT, image_features: list = None):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        if not image_features:
            img_data = []
            for split in dataset.splits:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                img_data.extend(load_obj_tsv(
                    os.path.join(imgfeat_dir, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                    topk=load_topk))
            import pdb; pdb.set_trace()
        else:
            img_data = image_features

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
            else:
                import pdb; pdb.set_trace()

        # print("Use %d data in torch dataset" % (len(self.data)))
        # print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()

        # Normalize the boxes (to 0 ~ 1)
        if 'normalized_boxes' not in img_info:
            boxes = img_info['boxes'].copy()
            img_h, img_w = img_info['img_h'], img_info['img_w']
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
        else:
            boxes = img_info['normalized_boxes'].copy()
        try:
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)
        except:
            import pdb; pdb.set_trace()
        assert obj_num == len(boxes) == len(feats)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path, human_readable: bool = False):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            results = []
            for ques_id, ans in quesid2ans.items():
                result = {
                    'question_id': ques_id,
                    'answer': ans
                }
                if human_readable:
                    question_datum = self.dataset.id2datum[ques_id]
                    result['question'] = question_datum
                results.append(result)
            json.dump(results, f, indent=4, sort_keys=True)


