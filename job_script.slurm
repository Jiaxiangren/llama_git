#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J rjx_job
#SBATCH -o ./rjx_job_ouput.txt
#SBATCH -e ./rjx_job_error.txt
#SBATCH -p trustlab
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:2

conda activate lora
# Use '&' to start the first job in the background
bash /work/cvp352/rjx_work/ACL/llama_git/auto_sum.sh