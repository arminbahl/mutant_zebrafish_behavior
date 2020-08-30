#!/bin/bash
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=52
#SBATCH --time=20:00:00 # Runtime
#SBATCH --mem=60000 # Total memory pool for one or more cores (see also --mem-per-cpu)
#SBATCH -o slurm_results_%A.out # Standard out goes to this file
#SBATCH -e slurm_results_%A.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL        # Send mail when process begins, fails, or finishes
#SBATCH --mail-user=arminbahl@fas.harvard.edu

module load GCCcore/8.2.0 CMake/3.14.4
module load Anaconda3/5.0.1-fasrc02

source activate py37

export ANTS_BIN_PATH="/n/home10/abahl/bin"
export CMTK_BIN_PATH="/n/home10/abahl/cmtk-build/bin"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=52
export CMTK_USE_THREADS=52
export CAIMAN_USE_THREADS=52
export MAIN_2P_DATA_PATH="/n/boslfs/LABS/engert_lab/arminbahl/2P Data"
export REFERENCE_BRAINS_PATH="/n/boslfs/LABS/engert_lab/arminbahl/zebrafish_atlases"
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="${PYTHONPATH}:/n/home10/abahl/python_projects/mutant_zebrafish_behavior/armin_analysis"

echo "Running Python on cluster...."
echo "$@"
hostname
date

srun python "$@"

echo "... Finished"
date
