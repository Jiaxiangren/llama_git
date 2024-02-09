#!/bin/bash
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -J rjx_job
#SBATCH -o ./rjx_job_ouput.txt
#SBATCH -e ./rjx_job_error.txt
#SBATCH -p trustlab
#SBATCH -t 48:00:00


# Use '&' to start the first job in the background
srun -n 1 bash /work/cvp352/rjx_work/ACL/llama_git/auto_1.sh &
srun -n 1 bash /work/cvp352/rjx_work/ACL/llama_git/auto_2.sh &
srun -n 1 bash /work/cvp352/rjx_work/ACL/llama_git/auto_3.sh &
srun -n 1 bash /work/cvp352/rjx_work/ACL/llama_git/auto_4.sh

# Use 'wait' as a barrier to collect both executables when they are done. If not the batch job will finish when the job2.batch program finishes and kill job1.batch if it is still running.
wait