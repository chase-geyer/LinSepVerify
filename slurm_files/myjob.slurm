#!/bin/bash

#SBATCH --job-name=linsepverify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:59:59
#SBATCH --mem=16000m
#SBATCH --mail-user=cjg8@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --partition=commons
#SBATCH --exclusive
#SBATCH --reservation=comp322

cd /home/$USER/or-research/LinSepVerify/ 

module load Julia # Load the Julia module
module load Gurobi # Load the Gurobi module

julia export_results.jl  # Run your Julia script