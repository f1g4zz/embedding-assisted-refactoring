#!/bin/bash
#BATCH --job-name=tesi_designite
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=ulow

# Entra nella cartella
cd /home/a.lanza-thesis/progetti_tesi

# Carica l'ambiente
source /home/a.lanza-thesis/anaconda3/etc/profile.d/conda.sh
conda activate designite_env

# Lancia usando il python specifico dell'ambiente
/home/a.lanza-thesis/.conda/envs/designite_env/bin/python -u run_designite_final.py