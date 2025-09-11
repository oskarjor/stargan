#!/bin/bash
#SBATCH --job-name="classifier_train"   # Sensible name for the job
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

python main_classifier.py \
    --model resnet18 \
    --image_size 128 \
    --log_dir classifier/logs \
    --model_save_dir classifier/models \
    --selected_attrs Young \
    --batch_size 256
