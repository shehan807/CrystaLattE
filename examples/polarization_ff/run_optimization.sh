#!/bin/bash

# Run polarizability optimization for acetic acid
# The script will optimize the 4 polarizability parameters to maximize R^2

echo "Starting polarizability optimization..."
echo "This will take approximately 15-20 minutes for 50 iterations"
echo ""

# Ensure we're in the right conda environment
micromamba activate om_p4_jax

# Run the optimization with default settings
# You can modify these parameters:
# --molecule: molecule name (default: acetic_acid)
# --distance-cutoff: only consider dimers with separation > this value in Angstroms (default: 12)
# --maxiter: maximum optimization iterations (default: 50)
# --method: optimization method (default: L-BFGS-B)

python optimize_polarizabilities.py \
    --molecule acetic_acid \
    --distance-cutoff 12 \
    --maxiter 50 \
    --method L-BFGS-B

echo ""
echo "Optimization complete!"
echo "Check the following files for results:"
echo "  - acetic_acid/acetic_acid_optimized.xml (optimized force field)"
echo "  - acetic_acid/acetic_acid_optimization_log.csv (optimization history)"
