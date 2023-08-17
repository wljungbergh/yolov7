SLURM_SCRIPT = """#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem 100G
#SBATCH --time 1-00:00:00
#SBATCH --output /workspaces/%u/yolov7logs/slurm-%j-{name}_{mode}_{iteration}-test.out
#SBATCH --partition zprod

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --pwd /workspaces/s0001387/repos/yolov7 \
    /staging/agp/geohe/containers/yolov7.sif \
    python -u test.py\
     --weights "/workspaces/s0001387/repos/yolov7/runs/train/{name}_{mode}_{iteration}/weights/best.pt"\
     --device 0\
     --batch-size 4\
     --data /workspaces/s0001387/repos/yolov7/data/zod_{mode}.yaml\
     --img 1920\
     --name {name}_{mode}_{iteration}\
     --task test\
     --verbose\
     --save-json\
     --exist-ok\
     """

import subprocess

N_ITERATIONS = 3
MODES = ["blur", "dnat", "original"]
NAME = "pretrained"


def main():
    i = 0
    for iter in range(N_ITERATIONS):
        for mode in MODES:
            script = SLURM_SCRIPT.format(
                name=NAME,
                mode=mode,
                iteration=iter,
                blur=mode,
            )
            slurm_script_name = f"scripts/{NAME}_{mode}_{iter}_test.sh"
            with open(slurm_script_name, "w") as f:
                f.write(script)
            subprocess.run(["sbatch", slurm_script_name])
            i += 1


if __name__ == "__main__":
    main()
