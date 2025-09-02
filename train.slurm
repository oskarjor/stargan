#!/bin/bash
#SBATCH --job-name="stargan_train"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1            # Number of GPUs
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-01:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/stargan/venv/bin/activate

python main.py --mode train --dataset CelebA --image_size 256 --c_dim 3 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs No_Beard Chubby Eyeglasses \
	       --batch_size 32 \
	       --use_tensorboard=True
