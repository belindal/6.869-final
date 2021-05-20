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
    image_set = set()
    removed = 0
    for pokemon in tqdm(glob.glob(os.path.join(raw_images_dir, "*"))):
        for image in glob.glob(os.path.join(pokemon, "*")):
            if not image.endswith('.jpg') and not image.endswith('.jpeg') and not image.endswith('.png'):
                removed += 1
                print(image)
                os.remove(image)
            else:
                image_name = ".".join(image.split(".")[0:-1])
                if image_name in image_set:
                    removed += 1
                    print(image)
                    os.remove(image)
                image_set.add(image_name)
    print(f"Removed {removed} images")


def get_side_by_side_ex(split, pos_pokemon, pos_image, valid_neg_pokemons, all_pokemon_name_to_imgs, qid, neg_pokemon=None, neg_image=None):
    """
    Returns example entry from putting positive and negative image side-by-side
    {
        "answer_type": "other", 
        "img_id": "c3po_r2d2", 
        "label": {
            "gold": 1, 
            "yellow": 0.3
        }, 
        "question_id": 1, 
        "question_type": "which side is the", 
        "sent": "Which side is the `pos_pokemon`?"
    }
    """
    if not neg_image:
        if not neg_pokemon:
            neg_pokemon = random.choice(valid_neg_pokemons)
        neg_image = random.choice(all_pokemon_name_to_imgs[neg_pokemon])

    # decide order of side by side
    image_order = [0,1]
    random.shuffle(image_order)
    image_list = [neg_image, pos_image]
    # make side-by-side image
    if not os.path.exists(os.path.join(raw_images_dir, split, 'side_by_side')):
        os.makedirs(os.path.join(raw_images_dir, split, 'side_by_side'), exist_ok=True)
    comb_imgpath = make_side_by_side_images(image_list[image_order[0]], image_list[image_order[1]], os.path.join(
        raw_images_dir, split, 'side_by_side',
        f'{pos_pokemon}_{".".join(os.path.split(pos_image)[-1].split(".")[0:-1])}_{neg_pokemon}_{".".join(os.path.split(neg_image)[-1].split(".")[0:-1])}.png',
    ))
    return {
        'img_id': comb_imgpath,
        'label': {'left' if image_order[0] == 1 else 'right': 1},
        'question_id': qid,
        'question_type': f'which side',
        'sent': f'Which side is the {pos_pokemon}?',
    }, neg_pokemon, neg_image


def make_qs_data(splits):
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
            pokemon_name = os.path.split(pokemon)[-1]
            if 'side_by_side' in pokemon_name: continue
            split_to_pokemon[split].append(pokemon_name)
            all_pokemon_names.append(pokemon_name)
            all_pokemon_name_to_imgs[pokemon_name] = list(glob.glob(os.path.join(pokemon, "*")))

    # blah blah
    for split in splits:
        output_dir = os.path.join(qs_dir, split)
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        for pokemon in tqdm(split_to_pokemon[split]):
            pos_images = all_pokemon_name_to_imgs[pokemon]
            assert len(pos_images) > 8
            # get random negative
            valid_neg_pokemons = [p for p in split_to_pokemon[split] if p is not pokemon]
            random.shuffle(pos_images)

            fewshot_data = {'train': [], 'eval': []}
            # make train data
            n_pos_images = 4
            qid = 0
            for i in range(n_pos_images):
                fewshot_data['train'].append({
                    'img_id': pos_images[i],
                    'label': {'yes': 1},
                    'question_id': qid,
                    'question_type': f'pos',
                    'sent': f'Is this a {pokemon}?',
                })
                qid += 1
            train_neg_pokemons = []
            train_neg_images = []
            n_neg_images = 4
            for _ in range(n_neg_images):
                neg_pokemon = random.choice(valid_neg_pokemons)
                train_neg_pokemons.append(neg_pokemon)
                neg_image = random.choice(all_pokemon_name_to_imgs[neg_pokemon])
                train_neg_images.append(neg_image)
                fewshot_data['train'].append({
                    'img_id': neg_image,
                    'label': {'no': 1},
                    'question_id': qid+1,
                    'question_type': f'neg',
                    'sent': f'Is this a {pokemon}?',
                })
                qid += 1
            fewshot_data['train'].append(get_side_by_side_ex(split, pokemon, pos_images[0], valid_neg_pokemons, all_pokemon_name_to_imgs, qid, neg_pokemon=neg_pokemon, neg_image=neg_image)[0])
            qid += 1

            for i in range(n_pos_images, 2*n_pos_images):
                # make eval data
                # Image generalization question
                fewshot_data['eval'].append({
                    'img_id': pos_images[i],
                    'label': {'yes': 1},
                    'question_id': qid,
                    'question_type': f'pos, image gen',
                    'sent': f'Is this a {pokemon}?',
                })
                qid += 1
            for _ in range(n_pos_images):
                fewshot_data['eval'].append({
                    'img_id': random.choice(all_pokemon_name_to_imgs[random.choice(valid_neg_pokemons)]),
                    'label': {'no': 1},
                    'question_id': qid,
                    'question_type': f'neg, image gen',
                    'sent': f'Is this a {pokemon}?',
                })
                qid += 1
            # Distinguishing question (no image generalization)
            for i in range(n_pos_images):
                comb_ex1, neg_pokemon1, neg_image1 = get_side_by_side_ex(split, pokemon, pos_images[i], valid_neg_pokemons, all_pokemon_name_to_imgs, qid, neg_pokemon=train_neg_pokemons[i], neg_image=train_neg_images[i])
                qid += 1
                comb_ex1['question_type'] += ', text gen'
                fewshot_data['eval'].append(comb_ex1)
                # Mutual exclusivity question
                mut_exc_ex = {
                    'img_id': comb_ex1['img_id'],
                    'label': {'left' if comb_ex1['label'].get('right', 0.0) > 0.5 else 'right': 1},
                    'question_id': qid,
                    'question_type': 'mutual exclusivity, text gen',
                    'sent': f'Which side is the {neg_pokemon1}?'
                }
                qid += 1
                fewshot_data['eval'].append(mut_exc_ex)

            # Distinguishing question + image generalization
            for i in range(n_pos_images, 2*n_pos_images):
                comb_ex2, neg_pokemon2, neg_image2 = get_side_by_side_ex(split, pokemon, pos_images[i], valid_neg_pokemons, all_pokemon_name_to_imgs, qid)
                qid += 1
                comb_ex2['question_type'] += ', image gen, text gen'
                fewshot_data['eval'].append(comb_ex2)
                # Mutual exclusivity question
                mut_exc_ex = {
                    'img_id': comb_ex2['img_id'],
                    'label': {'left' if comb_ex2['label'].get('right', 0.0) > 0.5 else 'right': 1},
                    'question_id': qid,
                    'question_type': 'mutual exclusivity, image gen, text gen',
                    'sent': f'Which side is the {neg_pokemon2}?'
                }
                qid += 1
                fewshot_data['eval'].append(mut_exc_ex)
            
            save_fn = os.path.join(output_dir, f'{pokemon}.json')
            json.dump(fewshot_data, open(save_fn, "w"), indent=4)


def make_side_by_side_images(left_imgpath, right_imgpath, comb_imgpath):
    try:
        imgs = [Image.open(i) for i in [left_imgpath, right_imgpath]]
        for i in range(len(imgs)):
            if imgs[i].mode != 'RGB':
                imgs[i] = imgs[i].convert("RGB")
            #     imgs[i].load()
            #     background = Image.new("RGB", imgs[i].size, (255, 255, 255))
            #     background.paste(imgs[i], mask=imgs[i].split()[3]) # 3 is the alpha channel
            #     imgs[i] = background
            # elif imgs[i].mode == “L”, “RGB” and “CMYK.”:

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
        import pdb; pdb.set_trace()
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


import shutil
def sample_dogs(dogs_dir: str="dogs/Images/n02099601-golden_retriever", amount: int=1, out_dir: str="dogs/subsample"):
    dogs = list(glob.glob(os.path.join(dogs_dir, "*.jpg")))
    chosen_dogs = random.sample(dogs, amount)
    for chosen_dog in chosen_dogs:
        shutil.move(chosen_dog, out_dir)

def make_dogs_side_by_side(pok_name, pok_image, dog_image, query_dog: bool, qid, dogs_dir = "dogs/subsample", out_imgs_dir = "img_dir/dogs_mut_excl"):
    # decide order of side by side
    image_order = [0,1]
    random.shuffle(image_order)
    image_list = [pok_image, dog_image]
    # make side-by-side image
    comb_imgpath = make_side_by_side_images(image_list[image_order[0]], image_list[image_order[1]], os.path.join(
        out_imgs_dir,
        f'{pok_name}_{".".join(os.path.split(pok_image)[-1].split(".")[0:-1])}_dog_{".".join(os.path.split(dog_image)[-1].split(".")[0:-1])}.png',
    ))
    return {
        'img_id': ".".join(os.path.split(comb_imgpath)[-1].split(".")[0:-1]),
        'label': {'left' if image_order[0] == 0 else 'right': 1},
        'question_id': qid,
        'question_type': f'which side',
        'sent': f'Which side is the {pok_name}?',
    }

def make_dogs_mut_excl_data():
    all_pokemon_name_to_imgs = {}
    all_pokemon_names = []

    for split in ['train', 'val', 'test']:
        for pokemon in glob.glob(os.path.join(raw_images_dir, split, "*")):
            pokemon_name = os.path.split(pokemon)[-1]
            if 'side_by_side' in pokemon_name: continue
            all_pokemon_names.append(pokemon_name)
            all_pokemon_name_to_imgs[pokemon_name] = list(glob.glob(os.path.join(pokemon, "*")))

    examples = []
    for d, dog_image in enumerate(glob.glob("dogs/subsample/*.jpg")):
        # choose random pokemon
        pok_name = random.choice(all_pokemon_names)
        pok_image = random.choice(all_pokemon_name_to_imgs[pok_name])
        ex = make_dogs_side_by_side(pok_name, pok_image, dog_image, query_dog=True, qid=d*2)
        examples.append(ex)
        examples.append({
            'img_id': ex['img_id'],
            'label': {'right' if ex['label'] == 'left' else 'right': 1},
            'question_id': d*2+1,
            'question_type': f'which side',
            'sent': f'Which side is the dog?',
        })
    json.dump(examples, open(f"data/vqa/dogs_mut_excl.json", 'w'), indent=4)

def main():
    # print("Filtering Images")
    # filter_image_data()
    # print("Splitting Images")
    # split_image_data()
    # print("Making Questions")
    # make_qs_data(splits=['test'])
    # print("Extracting FRCNN Features")
    # get_frcnn_features(splits_to_process=['test'])
    # sample_dogs()
    make_dogs_mut_excl_data()
    

if __name__ == "__main__":
    main()