# coding=utf-8
# Copyleft 2019 project LXRT.
from typing import List

import torch.nn as nn
import numpy as np
import torch

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU
from frcnn.modeling_frcnn import GeneralizedRCNN

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers, frcnn: GeneralizedRCNN = None, frcnn_cfg = None, num_features=2048):
        super().__init__()
        
        self.frcnn = frcnn
        self.frcnn_cfg = frcnn_cfg

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
        )
        hid_dim = self.lxrt_encoder.dim
        self.num_features = num_features
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def add_new_tokens(self, new_tokens: List[str]):
        new_tokens += [tok.lower() for tok in new_tokens]
        self.lxrt_encoder.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        # add new tokens to end
        self.lxrt_encoder.model.resize_token_embeddings(len(self.lxrt_encoder.tokenizer))

    def forward(
        self, feat=None, pos=None, sent=None,
        images=None, sizes=None, scales_yx=None,
    ):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        full_features = None
        if feat is None:
            assert images is not None and self.frcnn is not None
            output_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding=None,
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
            )
            pos = []
            feat = []
            for idx in range(len(images)):
                full_features, feat_list, info_list = self._process_features(output_dict, idx)
                boxes = info_list['normalized_boxes'].copy()
                np.testing.assert_array_less(boxes, 1+1e-5)
                np.testing.assert_array_less(-boxes, 0+1e-5)

                pos.append(boxes)
                feat.append(feat_list)
            feat = torch.stack(feat)
            pos = np.stack(pos)
            pos = torch.tensor(pos).to(feat.device)
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit, full_features

    def _process_features(self, features, index, confidence_threshold=0):
        feature_keys = [
            "obj_ids",
            "obj_probs",
            "attr_ids",
            "attr_probs",
            "boxes",
            "sizes",
            "original_sizes",
            "preds_per_image",
            "roi_features",
            "normalized_boxes",
        ]
        single_features = dict()

        for key in feature_keys:
            single_features[key] = features[key][index]

        confidence = confidence_threshold
        idx = 0
        while idx < single_features["obj_ids"].size()[0]:
            removed = False
            if (
                single_features["obj_probs"][idx] < confidence
                or single_features["attr_probs"][idx] < confidence
            ):
                single_features["obj_ids"] = torch.cat(
                    [
                        single_features["obj_ids"][0:idx],
                        single_features["obj_ids"][idx + 1 :],
                    ]
                )
                single_features["obj_probs"] = torch.cat(
                    [
                        single_features["obj_probs"][0:idx],
                        single_features["obj_probs"][idx + 1 :],
                    ]
                )
                single_features["attr_ids"] = torch.cat(
                    [
                        single_features["attr_ids"][0:idx],
                        single_features["attr_ids"][idx + 1 :],
                    ]
                )
                single_features["attr_probs"] = torch.cat(
                    [
                        single_features["attr_probs"][0:idx],
                        single_features["attr_probs"][idx + 1 :],
                    ]
                )
                single_features["boxes"] = torch.cat(
                    [
                        single_features["boxes"][0:idx, :],
                        single_features["boxes"][idx + 1 :, :],
                    ]
                )
                single_features["preds_per_image"] = (
                    single_features["preds_per_image"] - 1
                )
                single_features["roi_features"] = torch.cat(
                    [
                        single_features["roi_features"][0:idx, :],
                        single_features["roi_features"][idx + 1 :, :],
                    ]
                )
                single_features["normalized_boxes"] = torch.cat(
                    [
                        single_features["normalized_boxes"][0:idx, :],
                        single_features["normalized_boxes"][idx + 1 :, :],
                    ]
                )
                removed = True
            if not removed:
                idx += 1

        feat_list = single_features["roi_features"]

        boxes = single_features["boxes"][: self.num_features].cpu().numpy()
        num_boxes = self.num_features
        objects = single_features["obj_ids"][: self.num_features].cpu().numpy()
        probs = single_features["obj_probs"][: self.num_features].cpu().numpy()
        width = single_features["sizes"][1].item()
        height = single_features["sizes"][0].item()
        normalized_boxes = single_features['normalized_boxes'][: self.num_features].cpu().numpy()
        info_list = {
            "bbox": boxes,
            "num_boxes": num_boxes,
            "objects": objects,
            "cls_prob": probs,
            "image_width": width,
            "image_height": height,
            "normalized_boxes": normalized_boxes,
        }

        return single_features, feat_list, info_list

