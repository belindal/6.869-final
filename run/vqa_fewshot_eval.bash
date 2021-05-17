# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/fewshot_vqa.py \
    --train train --image_features data/vqa/fewshot_imgfeat \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --meta_lr 1e-4 --meta_epochs 10 --num_fewshot_updates 1 \
    --tqdm --output $output ${@:3}
