#!/bin/bash
#SBATCH --job-name=Job
#SBATCH --partition=general
#SBATCH --output=logs_predictions/out.%a.%N.%j.out
#SBATCH --error=logs_predictions/out.%a.%N.%j.err
##SBATCH --mail-user=
##SBATCH --mail-type=ALL
#SBATCH -n 1 # number of jobs
#SBATCH -c 1 # number of cpu
##SBATCH --array=1-24
#SBATCH --mem=5G

# command line process down here.

python 02_01_prediction_kmeans_clusters.py