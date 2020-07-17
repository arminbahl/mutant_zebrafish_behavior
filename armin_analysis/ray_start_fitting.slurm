#!/bin/bash
#SBATCH --job-name=test
#SBATCH -p shared # Partition to submit to
#SBATCH --nodes=10
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=50000 # Memory per node
#SBATCH -o slurm_results_%A.out # Standard out goes to this file
#SBATCH -e slurm_results_%A.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL        # Send mail when process begins, fails, or finishes
#SBATCH --mail-user=arminbahl@fas.harvard.edu

echo "Running Python on cluster (multiple nodes)...."
echo "$@"
hostname
date

worker_num=9 # Must be one less that the total number of nodes

module load Anaconda3/5.0.1-fasrc02
source activate py37

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 5
done

export PYTHONPATH="${PYTHONPATH}:/n/home10/abahl/python_projects/my_data_analyses/my_helpers"

python -u fit_integrator_model.py "$@"

echo "... Finished"
date
