#!/bin/bash
#SBATCH --job-name=kitt_gen_sem_pcd
#SBATCH --nodes=1
#SBATCH -A kvd@gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=19:59:00
#SBATCH --output=proj3d2d_%j.out
#SBATCH --error=proj3d2d_%j.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=10

module purge
conda deactivate
module load pytorch-gpu/py3/1.7.1
python $WORK/code/xmuda-extend/xmuda/common/data_gen/kitti_seg2d_to_pcd.py 
#
