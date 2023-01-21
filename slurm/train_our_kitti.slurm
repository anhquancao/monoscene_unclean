#!/bin/bash
#SBATCH --job-name=ablate_kitti
#SBATCH --nodes=1
#SBATCH -A kvd@gpu
#SBATCH --ntasks-per-node=1

#SBATCH --time=19:59:00
##SBATCH --time=49:59:00
##SBATCH --qos=qos_gpu-t4

#SBATCH --output=ablate_kitti_%j.out
#SBATCH --error=ablate_kitti_%j.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:4

##SBATCH -C v100-32g
##SBATCH --cpus-per-task=40

#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_p2


module purge
conda deactivate
module load pytorch-gpu/py3/1.7.1
python $WORK/code/xmuda-extend/xmuda/train_2d_proj3d2d.py batch_size=4 n_gpus=4 enable_log=true exp_prefix=ManualGroup16 project_scale=2 project_1_2=true project_1_4=true project_1_8=true project_1_16=false run=3 dataset=kitti project_scale=2 class_proportion_loss=true frustum_size=8 virtual_img=false context_prior=CRCP weight_decay=0 optimize_iou=true MCA_ssc_loss=true CE_relation_loss=true n_relations=16 corenet_proj=null
