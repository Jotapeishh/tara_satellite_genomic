#!/bin/bash
#SBATCH --job-name=ExtractSatData
#SBATCH --partition=general
#SBATCH --output=logs_models/out.%a.%N.%j.out
#SBATCH --error=logs_models/out.%a.%N.%j.err
##SBATCH --mail-user=
##SBATCH --mail-type=ALL
#SBATCH -n 1 # number of jobs
#SBATCH -c 1 # number of cpu
##SBATCH --array=1-24
#SBATCH --mem=5G

# command line process down here.

conda activate tara

python 00_00_extract_satellite_data.py "$1"