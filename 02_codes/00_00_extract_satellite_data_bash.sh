#!/bin/bash

for i in {1..50}
do
    sbatch sbatch_extract_sat_data.sh $i
done
