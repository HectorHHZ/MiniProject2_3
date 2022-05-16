#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=5:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=rtx8000
#SBATCH --job-name=block1


#module purge
#module load python/intel/3.8.6
#module load anaconda3/2020.07
#lsmodule load cuda/11.3.1

#conda init bash
module purge
eval "$(conda shell.bash hook)"
conda activate /scratch/zw2655/penv

#srun python3 train.py --name cifar10-100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --train_batch_size=64 --eval_batch_size = 64>> output.txt
#srun python train.py --name cifar10-100_500 --train_batch_size 1 --dataset cifar10 --fp16 --fp16_opt_level O2 >> output.txt
srun python3 train.py --name cifar10_Block1_part_single