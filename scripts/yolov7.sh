#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --time 14-00:00:00
#SBATCH --output /workspaces/%u/yolov7logs/slurm-%j-run.out
#SBATCH --partition zprod

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --env WANDB_API_KEY=12ac29ba00d9b55503af4f91224567f578727bc1 \
    --pwd /workspaces/s0001387/repos/yolov7 \
    /staging/agp/geohe/containers/yolov7.sif \
    python -u train_aux.py --workers 8\
     --weights "weights/yolov7-e6_training.pt"\
     --device 0\
     --adam\
     --batch-size 4\
     --cache-images\
     --data /workspaces/s0001387/repos/yolov7/data/zod_blur.yaml\
     --save_period 5\
     --img 1920\
     --cfg cfg/training/yolov7_zod.yaml\
     --name zod-blur-e6-pretrain\
     --hyp data/hyp.scratch.custom.yaml