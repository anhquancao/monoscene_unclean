#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH -A kvd@gpu
#SBATCH --ntasks-per-node=1
##SBATCH --time=99:59:00
#SBATCH --time=5:59:00
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err
#SBATCH --hint=nomultithread
##SBATCH --qos=qos_gpu-t4

#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu_p2

##SBATCH --cpus-per-task=10
##SBATCH -C v100-32g
#SBATCH --cpus-per-task=12

module purge
conda deactivate
module load pytorch-gpu/py3/1.7.1

python $WORK/code/xmuda-extend/xmuda/train_baseline.py batch_size=4 n_gpus=4 enable_log=true exp_prefix=DepthNoise4 dataset=kitti method=LMSCNet use_predicted_depth=true

