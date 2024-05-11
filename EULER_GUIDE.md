# To run on EULER

Login to Euler (you have to be on the ETH network or use a VPN):
```bash
ssh <username>@euler.ethz.ch
```
If it's the first time you are logging in, you will have to set up your account via some email verification.

## (Initial) Setup

(Optional) Generate an ssh key, so you will be able to push changes to the repository:
```bash
ssh-keygen -t rsa -b 4096
```

And add the public key to your github account:
```bash
cat ~/.ssh/id_rsa.pub
```
and insert in https://github.com/settings/keys (New SSH key)

Clone the repository:
```bash
git clone git@github.com:daniisler/dc_solubility_prediction.git
```

## Loading the modules

You might need to change to the new software stack:
```bash
source /cluster/apps/local/env2lmod.sh
```

To set the software stack to the new one permanently:
```bash
set_software_stack.sh new
```

Load the necessary modules (every time):
```bash
module load gcc/8.2.0 python/3.11.2
```

Install the necessary packages (only need to do this once):
```bash
pip install -r requirements.txt
```

## Start calculations

Except for maybe short tests DO NOT RUN CODE ON THE LOGIN NODE. Use the batch system:
```bash
sbatch --wrap="python <script>.py"
```
Optional tags for sbatch: See the [Euler documentation](https://scicomp.ethz.ch/wiki/Using_the_batch_system).

Example:
```bash
sbatch --ntasks=1 --cpus-per-task=96 --mem-per-cpu=5G --time 15:00:00 --wrap="python descriptor_calculation.py"
```

To check the status of your jobs:
```bash
squeue --all
```

The stdout of the jobs will be saved in the directory `slurm-<job_id>.out`.

## Useful commands

The output is usually very long, to not have the whole terminal filled with the output, you can use the `less` command:
```bash
less <file>
```
To scroll use the arrow keys, to exit press `q`.

To search for a specific string in the output:
```bash
grep <string> <file>
```

To see the last lines of the output:
```bash
tail -<number of lines> <file>
```

To see the first lines of the output:
```bash
head -<number of lines> <file>
```

To make quick changes to a file:
```bash
nano <file>
```
To save the changes press `Ctrl + O`, to exit press `Ctrl + X` and use the arrow keys to navigate.
