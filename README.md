# Learning to Learn Few-Shot VQA

## General 
The code requires **Python 3** and please install the Python dependencies with the command:
```bash
pip install -r requirements.txt
```

## LXMERT
### Checkpoint
My trained LXMERT checkpoints are available in `snap.zip`.

The finetuned (no metalearning) LXMERT is in `snap/vqa/vqa_lxr955/BEST.pth`.

To finetune your own checkpoint, refer to the README.md in the original LXMERT repositiory (https://github.com/airsplay/lxmert).

All other meta-learned checkpoints are under the subfolders in `snap/vqa/*`.


## Few-shot Concept Learning

### Data
You can download my few-shot data by running:
```bash
unzip PokemonData.zip
unzip fewshot.zip
unzip fewshot_imgfeat.zip
```

### Few-shot meta-evaluation
(Assumes GPU access)
Non meta-learned LXMERT
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_lxr955 --load snap/vqa/vqa_lxr955/BEST --num_fewshot_updates 5 --test {val|test}
```

Meta-learned full model
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon --load snap/vqa/vqa_fewshot_pokemon/BEST --num_fewshot_updates 10 --test {val|test}
```

Word embeddings only -- no added vocab
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon_toks_only_no_vocab --load snap/vqa/vqa_fewshot_pokemon_toks_only_no_vocab/BEST --test {val|test} --meta_word_embeds_only --learn_word_embeds_only --num_fewshot_updates 10 --lr 1e-2
```

Word embeddings only -- w/ added vocab
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon_meta_toks_only --load snap/vqa/vqa_fewshot_pokemon_meta_toks_only/BEST --add_pokemon_vocab --test {val|test} --meta_word_embeds_only --learn_word_embeds_only --lr 1e-2 --num_fewshot_updates 10
```

Answer classification head only
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon_anshead_only --load snap/vqa/vqa_fewshot_pokemon_anshead_only/BEST --test {val|test} --num_fewshot_updates 10
```
Add `--load_frcnn` features to each of vqa commands in order use the frcnn (instead of pre-loaded features)


### Few-shot training (metalearning)
(Assumes GPU access)
If you want to meta-learn yourself, run:
Top-Line result: Meta-learning full LXMERT
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon --load snap/vqa/vqa_lxr955/BEST --meta_epochs 50 --meta_lr 1e-4
```

Word embeddings only -- no added vocab
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon_toks_only_no_vocab --load snap/vqa/vqa_lxr955/BEST --meta_epochs 50 --meta_word_embeds_only --learn_word_embeds_only
```

Word embeddings only -- w/ added vocab
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon_meta_toks_only --load snap/vqa/vqa_lxr955/BEST --meta_epochs 50 --meta_word_embeds_only --learn_word_embeds_only --add_pokemon_vocab
```

Answer classification head only
```bash
bash run/vqa_fewshot_eval.bash 0 vqa_fewshot_pokemon_anshead_only --load snap/vqa/vqa_lxr955/BEST --meta_answer_head_only --meta_epochs 50
```

### Dogs vs. Pokemon
```bash
bash run/vqa_test.bash 0 vqa_dogs_results --load snap/vqa/vqa_lxr955/BEST --valid dogs_mut_excl --load_frcnn --test dogs_mut_excl --batchSize 1
```
