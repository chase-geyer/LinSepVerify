#!/bin/bash

#SBATCH --job-name=linsepverify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60000m
#SBATCH --time=01:00:00
#SBATCH --mail-user=cjg8@rice.edu # TODO: Input your net id
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --partition=scavenge
#SBATCH --exclusive
#SBATCH --reservation=comp322
 
cd /home/$USER/or-research/LinSepVerify/ # TODO: Change path to your hw 4 folder

module load Julia # Load the Julia module
module load Gurobi # Load the Gurobi module

julia export_results.jl  # Run your Julia script