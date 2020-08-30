srun -n 1 --pty -p test -t 5:00:00 --mem=50000 /bin/bash
module load Anaconda3/5.0.1-fasrc02
source activate py37
