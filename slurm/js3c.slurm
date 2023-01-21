#!/bin/bash
#SBATCH --job-name=JS3C
#SBATCH --nodes=1
#SBATCH -A kvd@gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=99:59:00
#SBATCH --output=js3c_%j.out
#SBATCH --error=js3c_%j.err
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t4

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu_p2

module purge
conda deactivate
module load pytorch-gpu/py3/1.3.1
cd $WORK/code/JS3C-Net
python train.py --gpu 0 --log_dir sem_pcd_v2 --config opt/JS3C_default_kitti.yaml
