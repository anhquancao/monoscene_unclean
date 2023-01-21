#!/bin/bash
#SBATCH --job-name=IoU
#SBATCH --nodes=1
#SBATCH -A kvd@gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=19:59:00
#SBATCH --output=proj3d2d_%j.out
#SBATCH --error=proj3d2d_%j.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu_p2

##SBATCH -C v100-32g
##SBATCH --cpus-per-task=20

module purge
conda deactivate
module load pytorch-gpu/py3/1.7.1
python $WORK/code/xmuda-extend/xmuda/train_2d_proj3d2d.py batch_size=4 n_gpus=2 enable_log=true exp_prefix=InvFreqCLassRel_Rerun project_1_2=true project_1_4=true project_1_8=true project_1_16=true run=1 dataset=NYU class_proportion_loss=false kl_w=1 force_empty_w=1 class_relation_weights=inv_freq context_prior=CRCP MCA_ssc_loss=true
#
