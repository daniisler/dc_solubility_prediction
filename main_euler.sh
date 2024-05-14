#!/bin/bash
#SBATCH --job-name=dc_solubility    #Name of the job
#SBATCH --ntasks=1                  #Requesting 1 node (is always 1)
#SBATCH --cpus-per-task=128         #Requesting 128 CPU
#SBATCH --mem-per-cpu=1G            #Requesting 1 Gb memory per core
#SBATCH --time=50:00:00             #Requesting 50 hours running time
#SBATCH --output logs/main.log      #Log-file


#Change to the new software stack
source /cluster/apps/local/env2lmod.sh

#Load the needed modules
module load gcc/8.2.0 python/3.11.2 eth_proxy

#Set environment variables
export NUMEXPR_MAX_THREADS=128
export NUM_WORKERS=100

#Run the python script
python main.py