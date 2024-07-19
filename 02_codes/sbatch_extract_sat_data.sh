#!/bin/bash
#SBATCH --job-name=ExtractSatData
#SBATCH --partition=general
#SBATCH --output=logs/extract_sat.%a.%N.%j.out
#SBATCH --error=logs/extract_sat.%a.%N.%j.err
##SBATCH --mail-user=
##SBATCH --mail-type=ALL
#SBATCH -n 1 # number of jobs
#SBATCH -c 1 # number of cpu
##SBATCH --array=1-24
#SBATCH --mem=2G

python 00_00_extract_satellite_data_test.py $1
