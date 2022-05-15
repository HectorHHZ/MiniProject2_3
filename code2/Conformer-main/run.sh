#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=2                # uses 2 compute core per task
#SBATCH --time=40:00:00                  # 20 hours is enough
#SBATCH --mem=100GB
#SBATCH --gres=gpu:rtx8000:4             # distribute
#SBATCH --partition=rtx8000
#SBATCH --job-name=Conformer             # job name

#module purge
#module load python/intel/3.8.6
#module load anaconda3/2020.07
#lsmodule load cuda/11.3.1

#conda init bash
module purge
eval "$(conda shell.bash hook)"
conda activate /scratch/zw2655/penv     # if needed, switch to your own env

export CUDA_VISIBLE_DEVICES=0,1,2,3     # make sure your device rank correct
#OUTPUT='./output/Conformer_small_patch16_batch_1024_lr1e-3_300epochs'
OUTPUT='./output/Conformer_small_patch16_batch_1024_lr1e-3_100epochs_bz_16_CIFAR10'
#OUTPUT='./output/Conformer_small_patch16_batch_1024_lr1e-3_300epochs_bz128_CIFAR100'

python -m torch.distributed.launch --master_port 50130 --nproc_per_node=4 --use_env main.py \
                                  --model Conformer_small_patch16 \
                                   --data-set CIFAR10 \
                                   --batch-size 16 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path /scratch/zw2655/project23/Conformer-main/CIFAR10/ \
                                   --output_dir ${OUTPUT} \
                                   --epochs 100

# Inference
#CUDA_VISIBLE_DEVICES=0, python main.py  --model Conformer_tiny_patch16 --eval --batch-size 64 \
#                --input-size 224 \
#                --data-set IMNET \
#                --num_workers 4 \
#                --data-path ../ImageNet_ILSVRC2012/ \
#                --epochs 100 \
#                --resume ../Conformer_tiny_patch16.pth