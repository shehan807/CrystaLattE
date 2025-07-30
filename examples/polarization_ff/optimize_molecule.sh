#!/bin/bash

# Flexible polarizability optimization script
# Usage: ./optimize_molecule.sh [molecule_name] [maxiter]
# Examples:
#   ./optimize_molecule.sh acetic_acid 50
#   ./optimize_molecule.sh pyrazole 100
#   ./optimize_molecule.sh  (defaults to pyrazole with 50 iterations)

# Set defaults
MOLECULE=${1:-pyrazole}
MAXITER=${2:-50}
CUTOFF=${3:-12}

echo "=========================================="
echo "Polarizability Optimization"
echo "=========================================="
echo "Molecule: $MOLECULE"
echo "Max iterations: $MAXITER"
echo "Distance cutoff: $CUTOFF Å"
echo "=========================================="
echo ""

# Ensure we're in the right conda environment
if command -v micromamba &> /dev/null; then
    micromamba activate om_p4_jax
elif command -v conda &> /dev/null; then
    conda activate om_p4_jax
else
    module load anaconda3
    conda activate om_p4_jax
fi

# First, update summarize_results.sh to use the correct molecule
sed -i.bak "s/MOLECULES=([^)]*)/MOLECULES=(\"$MOLECULE\")/" summarize_results.sh

echo "Starting optimization..."
start_time=$(date +%s)

# Run the optimization
python optimize_polarizabilities.py \
    --molecule $MOLECULE \
    --distance-cutoff $CUTOFF \
    --maxiter $MAXITER \
    --method L-BFGS-B

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Optimization complete in $duration seconds!"
echo "=========================================="
echo "Results saved to:"
echo "  - $MOLECULE/${MOLECULE}_optimized.xml (optimized force field)"
echo "  - $MOLECULE/${MOLECULE}_optimization_log.csv (optimization history)"
echo ""
echo "To visualize the improvement, run:"
echo "  python plot_U_ind_comp.py -i $MOLECULE/$MOLECULE.csv -o $MOLECULE/${MOLECULE}_final.png --distance_cutoff $CUTOFF"