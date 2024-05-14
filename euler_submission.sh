#!/bin/bash
#SBATCH --job-name=dc_solubility        #Name of the job
#SBATCH --ntasks=1            #Requesting 1 node (is always 1)
#SBATCH --cpus-per-task=128     #Requesting 1 CPU
#SBATCH --mem-per-cpu=1G      #Requesting 4 Gb memory per core
#SBATCH --time=4:00:00        #Requesting 4 hours running time
#SBATCH --output logs/main.log      #Log


#Change to the new software stack
source /cluster/apps/local/env2lmod.sh

#Load the needed modules
module load gcc/8.2.0 python/3.11.2 eth_proxy

#The bcftools command
python main.py