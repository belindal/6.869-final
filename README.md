# Learning to Learn Few-Shot VQA

### General 
The code requires **Python 3** and please install the Python dependencies with the command:
```bash
pip install -r requirements.txt
```

By the way, a Python 3 virtual environment could be set up and run with:
```bash
virtualenv name_of_environment -p python3
source name_of_environment/bin/activate
```
### VQA
#### Fine-tuning
1. Please make sure the LXMERT pre-trained model is either [downloaded](#pre-trained-models) or [pre-trained](#pre-training).

2. Download the re-distributed json files for VQA 2.0 dataset. The raw VQA 2.0 dataset could be downloaded from the [official website](https://visualqa.org/download.html).
    ```bash
    mkdir -p data/vqa
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
    ```
3. Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (VQA 2.0 is collected on MS COCO dataset).
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
    ```bash
    mkdir -p data/mscoco_imgfeat
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
    ```

4. Before fine-tuning on whole VQA 2.0 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `vqa_lxr955_tiny` is the name of this experiment.
    ```bash
    bash run/vqa_finetune.bash 0 vqa_lxr955_tiny --tiny
    ```
5. If no bug came out, then the model is ready to be trained on the whole VQA corpus:
    ```bash
    bash run/vqa_finetune.bash 0 vqa_lxr955
    ```
It takes around 8 hours (2 hours per epoch * 4 epochs) to converge. 
The **logs** and **model snapshots** will be saved under folder `snap/vqa/vqa_lxr955`. 
The validation result after training will be around **69.7%** to **70.2%**. 


Make data and get FRCNN features
```bash
python make_data.py
python frcnn/extract_features_frcnn.py --image_dir img_dir --num_features 2048 --visualize True
python frcnn/visualize.py --image_path img_dir/001Bulbasaur.png --features_path frcnn_output/001Bulbasaur_full.npy
python frcnn/visualize.py --image_path img_dir/ --features_path frcnn_output/ 
```

```bash
bash run/vqa_test.bash 0 vqa_lxr955_results --load snap/vqa/vqa_lxr955/BEST --interact

bash run/vqa_finetune.bash 9 vqa_fewshot --load_frcnn --train fewshot_train --valid fewshot_train --load snap/vqa/vqa_lxr955/BEST --batchSize 1 --epochs 1   
bash run/vqa_test.bash 9 vqa_fewshot --load_frcnn --load snap/vqa/vqa_fewshot/BEST --interact
```

Few-shot training (metalearning) and eval
```bash
bash run/vqa_fewshot_eval.bash 11 vqa_fewshot_pokemon --load snap/vqa/vqa_lxr955/BEST --meta_epochs 50
bash run/vqa_fewshot_eval.bash 9 vqa_fewshot_pokemon_add_toks --load snap/vqa/vqa_lxr955/BEST --meta_epochs 50 --add_pokemon_vocab
bash run/vqa_fewshot_eval.bash 9 vqa_fewshot_pokemon --load snap/vqa/vqa_lxr955/BEST --test val

bash run/vqa_fewshot_eval.bash 11 vqa_fewshot_pokemon --load snap/vqa/vqa_lxr955/BEST --meta_epochs 50 --meta_lr 1e-3
bash run/vqa_fewshot_eval.bash 9 vqa_fewshot_pokemon_train_toks_only --load snap/vqa/vqa_lxr955/BEST --add_pokemon_vocab --meta_word_embeds_only --meta_epochs 50 --meta_lr 1e-3
```
Add `--load_frcnn` features to each of vqa commands in order use the frcnn (instead of pre-loaded features)

Expected Results (old dataset)
| | No training | + Meta-learning |
|---| ---- | ---- |
| Avg. Valid Support Score | 82.7 | 86.7 |
| Avg. Valid Query Score | 59 | 66 |

New dataset
===
Average Support Score: 0.98
Average Query Score: 0.536
Average Support Score: 1.0
Average Query Score: 0.616
===
Average Support Score: 0.96
Average Query Score: 0.544
Average Support Score: 0.98
Average Query Score: 0.6080000000000001

- Metalearning saved in `snap/vqa/vqa_fewshot_pokemon_fsupdates1/BEST`

Expected Results (new dataset)