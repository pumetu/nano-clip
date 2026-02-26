#!/bin/bash

student=ViT-T-16
teacher=hf-hub:apple/DFN2B-CLIP-ViT-L-14
name=debug
loss=clipkd

ml cuda12.9/toolkit/12.9.1 
ml cuda12.9/blas/12.9.1
ml cuda12.9/fft/12.9.1
ml nccl2-cuda12.9-gcc/2.28.3

cd /scratch_aisg/patomp/nano-clip

uv run torchrun --nproc-per-node 8 -m open_clip_train.main -- \
    --train-data '/scratch_aisg/patomp/data/datacomp-small/shards/{00000000..00001287}.tar' \
    --train-num-samples 12800000 \
    --dataset-type webdataset \
    --logs /scratch_aisg/patomp/nano-clip/models \
    --warmup 2000 \
    --batch-size 2048 \
    --epochs 100 \
    --lr 5e-4 \
    --workers 16 \
    --imagenet-val /scratch_aisg/patomp/data/imagenet/val_blurred \
    --zeroshot-frequency 1 \
    --save-frequency 10 \
    --model $student \
    --distill-model $teacher \
    --distill-loss $loss \
    --report-to None \
    --grad-clip-norm 10.0 \
    --name $name