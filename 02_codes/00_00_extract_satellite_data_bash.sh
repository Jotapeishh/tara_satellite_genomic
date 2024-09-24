#!/bin/bash

for i in {1..50}
do
    sbatch 00_00_sbatch_extract_sat_data.sh $i
done
