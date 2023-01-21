#!/bin/bash
#SBATCH --job-name=IoU
#SBATCH --nodes=1
#SBATCH -A kvd@gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=19:59:00
#SBATCH --output=proj3d2d_%j.out
#SBATCH --error=proj3d2d_%j.err
#SBATCH --hint=nomultithread

##SBATCH --gres=gpu:2
#SBATCH --gres=gpu:1

##SBATCH --cpus-per-task=6
##SBATCH --partition=gpu_p2

##SBATCH -C v100-32g
##SBATCH --cpus-per-task=20
#SBATCH --cpus-per-task=10

module purge
conda deactivate
module load pytorch-gpu/py3/1.7.1
#python $WORK/code/xmuda-extend/xmuda/train_2d_proj3d2d.py batch_size=4 n_gpus=2 enable_log=true exp_prefix=v8_MegaCP_MulMask255_multilabel project_1_2=true project_1_4=true project_1_8=true project_1_16=false run=1 dataset=NYU class_proportion_loss=true kl_w=1 force_empty_w=1 context_prior=CRCP empty_multiplier=1.0
CUDA_VISIBLE_DEVICES=0 python $WORK/code/semantic-segmentation/demo_folder.py --snapshot $WORK/code/semantic-segmentation/pretrained_models/kitti_best.pth
#
