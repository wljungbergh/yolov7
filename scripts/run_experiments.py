SLURM_SCRIPT = """#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem 640G
#SBATCH --time 14-00:00:00
#SBATCH --output /workspaces/%u/yolov7logs/slurm-%j-{name}_{mode}_{iteration}.out
#SBATCH --partition zprod

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --pwd /workspaces/s0001387/repos/yolov7 \
    --env WANDB_API_KEY=12ac29ba00d9b55503af4f91224567f578727bc1 \
    /staging/agp/geohe/containers/yolov7.sif \
    python -u train_aux.py --workers 8\
     --weights "weights/yolov7-e6_training.pt"\
     --device 0\
     --adam\
     --batch-size 4\
     --cache-images\
     --data /workspaces/s0001387/repos/yolov7/data/zod_{mode}.yaml\
     --save_period 3\
     --img 1920\
     --cfg cfg/training/yolov7_zod.yaml\
     --name {name}_{mode}_{iteration}\
     --hyp data/hyp.scratch.custom.yaml"""

import subprocess

N_ITERATIONS = 3
MODES = ["blur", "dnat", "original"]
NAME = "pretrained"

ALL_NODES = [f"hal-gn{i:02}" for i in range(1, 21)]
POSSIBLE_NODES = [
    ("hal-gn02", "hal-gn03"),
    ("hal-gn04", "hal-gn06"),
    ("hal-gn07", "hal-gn08"),
    ("hal-gn09", "hal-gn10"),
    ("hal-gn11", "hal-gn12"),
    ("hal-gn13", "hal-gn14"),
    ("hal-gn15", "hal-gn16"),
    ("hal-gn17", "hal-gn18"),
    ("hal-gn19", "hal-gn20"),
]


def main():
    i = 0
    for iter in range(N_ITERATIONS):
        for mode in MODES:
            excluded_nodes = ",".join(
                f for f in ALL_NODES if f not in POSSIBLE_NODES[i]
            )
            script = SLURM_SCRIPT.format(
                name=NAME,
                mode=mode,
                iteration=iter,
                blur=mode,
            )
            slurm_script_name = f"scripts/{NAME}_{mode}_{iter}.sh"
            with open(slurm_script_name, "w") as f:
                f.write(script)
            subprocess.run(["sbatch", slurm_script_name])
            i += 1


if __name__ == "__main__":
    main()
