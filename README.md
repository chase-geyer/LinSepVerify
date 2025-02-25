# LinSepVerify

A Julia-based verification framework for neural networks using linear separable activation functions.

## Prerequisites

- Julia 1.8+
- Gurobi Optimizer
- Required Julia packages:
  ```julia
  using Pkg
  Pkg.add(["JuMP", "Gurobi", "Pickle", "Images", "Suppressor", "Profile", "LinearAlgebra"])
  ```

## Project Structure

```
  LinSepVerify/
├── export_files/           # Main execution files
│   ├── exact_verify.jl     # Main verification script
│   ├── export_m_sol.jl     # Big-M method implementation
│   ├── export_poly_sol.jl  # DeepPoly method implementation
│   └── export_results.jl   # Cayley method implementation
├── models/                 # Neural network model files
├── imgs/                   # Test images and labels
└── slurm_files/           # SLURM job submission scripts
```

## Running the Verification Scripts

There are three main verification methods implemented in Table 2

### 1. Big-M Method

```bash
cd export_files
julia export_m_sol.jl
```

### 2. Cayley Method

```bash
cd export_files
julia export_cayley_sol.jl
```

### 1. Big-M Method

```bash
cd export_files
julia export_m_sol.jl
```

## Output

Results are saved in:

big_m_results - Big-M method results
deep_poly_results - DeepPoly method results
cayley_outputs - Cayley method results

Each folder contains:

time_values/ - Execution time data
objective_gaps/ - Optimization gaps and objectives
