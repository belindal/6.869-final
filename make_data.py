import os
import shutil
import glob
import random
import csv
import sys
from tqdm import tqdm
import json
from frcnn.extract_features_frcnn import FeatureExtractor
from frcnn.extraction_utils import chunks, get_image_files
import copy
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from frcnn.visualize import visualize_images
import numpy as np

random.seed(42)


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

raw_images_dir = "./PokemonData"
image_ft_dir = "./data/vqa/fewshot_imgfeat"
qs_dir = "./data/vqa/fewshot"

splits = ['train', 'val', 'test']

def split_image_data():
    pokemons = list(glob.glob(f'{raw_images_dir}/*'))
    print(pokemons)
    for split in splits:
        if not os.path.exists(f'{raw_images_dir}/{split}'):
            os.makedirs(f'{raw_images_dir}/{split}', exist_ok=True)
        else:
            pokemons.remove(f'{raw_images_dir}/{split}')

    assert len(pokemons) == 150

    split_data = {
        'train': pokemons[:100],
        'val': pokemons[100:125],
        'test': pokemons[125:],
    }

    for split in split_data:
        for folder in split_data[split]:
            shutil.move(folder, f'{raw_images_dir}/{split}')


def filter_image_data():
    # remove images we can't process
    removed = 0
    for pokemon in tqdm(glob.glob(os.path.join(raw_images_dir, "*"))):
        for image in glob.glob(os.path.join(pokemon, "*")):
            if not image.endswith('.jpg') and not image.endswith('.jpeg') and not image.endswith('.png'):
                removed += 1
                print(image)
                os.remove(image)
    print(f"Removed {removed} images")


def make_qs_data():
    """
    {
        "answer_type": "other", 
        "img_id": "c3po_r2d2", 
        "label": {
            "gold": 1, 
            "yellow": 0.3
        }, 
        "question_id": 1, 
        "question_type": "what color is the", 
        "sent": "What color is the wug?"
    }
    """
    split_to_pokemon = {}
    all_pokemon_names = []
    # [*list(glob.glob(os.path.join(raw_images_dir, split))) for split in splits]
    # all_pokemon_names = [os.path.split(pokemon)[-1] for pokemon in all_pokemon_names]
    all_pokemon_name_to_imgs = {}

    for split in splits:
        split_to_pokemon[split] = []
        for pokemon in glob.glob(os.path.join(raw_images_dir, split, "*")):
            if pokemon == 'side_by_side': continue
            pokemon_name = os.path.split(pokemon)[-1]
            split_to_pokemon[split].append(pokemon_name)
            all_pokemon_names.append(pokemon_name)
            all_pokemon_name_to_imgs[pokemon_name] = list(glob.glob(os.path.join(pokemon, "*")))

    # blah blah
    for split in splits:
        output_dir = os.path.join(qs_dir, split)
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        for pokemon in tqdm(split_to_pokemon[split]):
            pos_images = all_pokemon_name_to_imgs[pokemon]
            # get random negative
            valid_neg_pokemons = [p for p in split_to_pokemon[split] if p is not pokemon]
            neg_pokemon = random.choice(valid_neg_pokemons)
            neg_images = all_pokemon_name_to_imgs[neg_pokemon]
            random.shuffle(pos_images)
            random.shuffle(neg_images)

            # TODO make side-by-side image
            image_order = [0,1]
            random.shuffle(image_order)
            image_list = [neg_images[0], pos_images[0]]
            if not os.path.exists(os.path.join(raw_images_dir, split, 'side_by_side')):
                os.makedirs(os.path.join(raw_images_dir, split, 'side_by_side'), exist_ok=True)
            comb_imgpath = make_side_by_side_images(image_list[image_order[0]], image_list[image_order[1]], os.path.join(
                raw_images_dir, split, 'side_by_side',
                f'{pokemon}_{os.path.split(pos_images[0])[-1].split(".")[0]}_{neg_pokemon}_{os.path.split(neg_images[0])[-1].split(".")[0]}.png',
            ))

            fewshot_data = {'train': [], 'eval': []}
            # make train data
            fewshot_data['train'].append({
                'img_id': pos_images[0],
                'label': {'yes': 1},
                'question_id': 1,
                'question_type': f'Is this a',
                'sent': f'Is this a {pokemon}?',
            })
            fewshot_data['train'].append({
                'img_id': pos_images[0],
                'label': {'no': 1},
                'question_id': 2,
                'question_type': f'Is this a',
                'sent': f'Is this a {random.choice(valid_neg_pokemons)}?',
            })
            fewshot_data['train'].append({
                'img_id': random.choice(all_pokemon_name_to_imgs[random.choice(valid_neg_pokemons)]),
                'label': {'no': 1},
                'question_id': 3,
                'question_type': f'Is this a',
                'sent': f'Is this a {pokemon}?',
            })
            # make eval data
            fewshot_data['eval'].append({
                'img_id': pos_images[1],
                'label': {'yes': 1},
                'question_id': 4,
                'question_type': f'Is this a',
                'sent': f'Is this a {pokemon}?',
            })
            fewshot_data['eval'].append({
                'img_id': pos_images[1],
                'label': {'no': 1},
                'question_id': 5,
                'question_type': f'Is this a',
                'sent': f'Is this a {random.choice(valid_neg_pokemons)}?',
            })
            fewshot_data['eval'].append({
                'img_id': random.choice(all_pokemon_name_to_imgs[random.choice(valid_neg_pokemons)]),
                'label': {'no': 1},
                'question_id': 6,
                'question_type': f'Is this a',
                'sent': f'Is this a {pokemon}?',
            })
            if comb_imgpath:
                fewshot_data['eval'].append({
                    'img_id': comb_imgpath,
                    'label': {'left' if image_order[0] == 1 else 'right': 1},
                    'question_id': 7,
                    'question_type': f'Which side is the',
                    'sent': f'Which side is the {pokemon}?',
                })
            save_fn = os.path.join(output_dir, f'{pokemon}.json')
            json.dump(fewshot_data, open(save_fn, "w"), indent=4)


def make_side_by_side_images(left_imgpath, right_imgpath, comb_imgpath):
    try:
        imgs = [Image.open(i) for i in [left_imgpath, right_imgpath]]
        for i in range(len(imgs)):
            if imgs[i].mode == 'RGBA':
                imgs[i].load()
                background = Image.new("RGB", imgs[i].size, (255, 255, 255))
                background.paste(imgs[i], mask=imgs[i].split()[3]) # 3 is the alpha channel
                imgs[i] = background

        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack([np.asarray( i.resize(min_shape)) for i in imgs])

        # save that beautiful picture
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(comb_imgpath)

        """
        # for a vertical stacking it is simple: use vstack
        imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        imgs_comb = PIL.Image.fromarray( imgs_comb)
        imgs_comb.save( 'Trifecta_vertical.jpg' )
        """
        return comb_imgpath
    except:
        print(left_imgpath)
        print(right_imgpath)
        return None


def get_frcnn_features(splits_to_process: list = ['train', 'val', 'test']):
    # FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf", "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
    fe = FeatureExtractor()
    fe.args.num_features = 2048
    fe.args.visualize = True
    print(fe.frcnn.device)
    for split in splits_to_process:
        finished = 0
        failed = 0
        failedNames = []
        for pokemon in tqdm(glob.glob(os.path.join(raw_images_dir, split, "*"))):
            pokemon_name = os.path.split(pokemon)[-1]
            output_dir = os.path.join(image_ft_dir, split, pokemon_name)
            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            fe.args.output_folder = output_dir

            image_files = list(glob.glob(os.path.join(pokemon, "*")))

            file_names = copy.deepcopy(image_files)

            total = len(file_names)
            for chunk, begin_idx in chunks(image_files, fe.args.batch_size):
                try:
                    features = fe.get_frcnn_features(chunk)
                    for idx, file_name in enumerate(chunk):
                        full_features, feat_list, info_list = fe._process_features(
                            features, idx
                        )
                        assert 'normalized_boxes' in info_list
                        info_list['img_id'] = file_names[begin_idx + idx]
                        full_features = {ft_name: full_features[ft_name].cpu() for ft_name in full_features}
                        fe._save_feature(
                            file_names[begin_idx + idx],
                            full_features,
                            feat_list.cpu(),
                            info_list,
                        )
                    finished += len(chunk)
                except Exception:
                    failed += len(chunk)
                    for idx, file_name in enumerate(chunk):
                        failedNames.append(file_names[begin_idx + idx])
                    print("message")
        if fe.args.partition is not None:
            print("Partition " + str(fe.args.partition) + " done.")
        print("Failed: " + str(failed))
        print("Failed Names: " + str(failedNames))



def main():
    # print("Filtering Images")
    # filter_image_data()
    # print("Splitting Images")
    # split_image_data()
    # print("Making Questions")
    # make_qs_data()
    print("Extracting FRCNN Features")
    get_frcnn_features(splits_to_process=['test'])

if __name__ == "__main__":
    main()