# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List, Optional, Tuple

import argparse
import numpy as np
import os
import torch
import torchvision
from tqdm import tqdm
from mmf.datasets.processors.frcnn_processor import img_tensorize
from frcnn.visualizing_image import SingleImageViz
from PIL import Image
import glob


def visualize_images(
    images: List[Any], size: Optional[Tuple[int, int]] = (224, 224), savepath=None, *args, **kwargs
):
    """Visualize a set of images using torchvision's make grid function. Expects
    PIL images which it will convert to tensor and optionally resize them. If resize is
    not passed, it will only accept a list with single image

    Args:
        images (List[Any]): List of images to be visualized
        size (Optional[Tuple[int, int]], optional): Size to which Images can be resized.
            If not passed, the function will only accept list with single image.
            Defaults to (224, 224).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Visualization tools require matplotlib. "
            + "Install using pip install matplotlib."
        )
        raise

    transform_list = []

    assert (
        size is not None or len(images) == 1
    ), "If size is not passed, only one image can be visualized"

    if size is not None:
        transform_list.append(torchvision.transforms.Resize(size=size))

    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)

    img_tensors = torch.stack([transform(image) for image in images])
    grid = torchvision.utils.make_grid(img_tensors, *args, **kwargs)

    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    if savepath:
        plt.savefig(savepath)
    # if savepath:
    #     plt.imsave(savepath, grid.permute(1, 2, 0))


def visualize_frcnn_features(
    image_path: str, features_path: str, out_path: str, objids: List[str], attrids: List[str]
):
    img = img_tensorize(image_path)

    output_dict = np.load(features_path, allow_pickle=True).item()
    id2obj = ['unknown' for _ in range(output_dict['obj_ids'].max()+1)]
    id2attr = ['unknown' for _ in range(output_dict['attr_ids'].max()+1)]
    # objids = [str(objid.item()) for objid in output_dict['obj_ids']]
    # attrids = [str(attrid.item()) for attrid in output_dict['attr_ids']]

    frcnn_visualizer = SingleImageViz(img, id2obj=id2obj, id2attr=id2attr)
    frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"),
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_probs"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_probs"),
    )

    height, width, channels = img.shape

    buffer = frcnn_visualizer._get_buffer()
    array = np.uint8(np.clip(buffer, 0, 255))

    image = Image.fromarray(array)

    visualize_images([image], (height, width), out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", default="img_dir", type=str, help="Image directory or file"
    )
    parser.add_argument(
        "--features_path", default="frcnn_output", type=str, help="FRCNN features directory or file"
    )
    parser.add_argument(
        "--out_path", default="frcnn_vis", type=str, help="Visualized features directory or file"
    )
    args = parser.parse_args()

    image_path = args.image_path
    features_path = args.features_path
    out_path = args.out_path

    if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
        # is directory
        image_files = glob.glob(os.path.join(image_path, "*.png"))
        image_files.extend(glob.glob(os.path.join(image_path, "*.jpg")))
        image_files.extend(glob.glob(os.path.join(image_path, "*.jpeg")))
        for image_fn in tqdm(image_files):
            features_fn = os.path.join(features_path, f"{os.path.split(image_fn)[-1].split('.')[0]}_full.npy")
            out_fn = os.path.join(out_path, f"{os.path.split(image_fn)[-1].split('.')[0]}.png")
            visualize_frcnn_features(image_fn, features_fn, out_fn, None, None)
    else:
        visualize_frcnn_features(image_path, features_path, out_path, None, None)
