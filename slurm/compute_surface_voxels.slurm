#!/bin/bash
#SBATCH --job-name=ratio
#SBATCH --nodes=1
#SBATCH -A kvd@gpu
#SBATCH --ntasks-per-node=1
##SBATCH --time=99:59:00
#SBATCH --time=19:59:00
#SBATCH --output=ratio_%j.out
#SBATCH --error=ratio_%j.err
#SBATCH --hint=nomultithread
##SBATCH --qos=qos_gpu-t4

#SBATCH --gres=gpu:1

##SBATCH --cpus-per-task=3
##SBATCH --partition=gpu_p2

##SBATCH --cpus-per-task=10
##SBATCH -C v100-32g
#SBATCH --cpus-per-task=10

module purge
conda deactivate
module load pytorch-gpu/py3/1.7.1
python $WORK/code/xmuda-extend/xmuda/common/data_gen/surface_voxels_ratio.py

