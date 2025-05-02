#!/bin/bash

# Define the list of molecules and distance cutoffs
MOLECULES=("acetic_acid" "imidazole" "pyrazole" "pyrazine") # You can replace this with ("imidazole" "pyrazole" "pyrazine" "acetic_acid")
DISTANCE_CUTOFFS=(2 12)  # Example values, modify as needed

# Export the molecules list as an environment variable
export MOLECULES_LIST="$(IFS=,; echo "${MOLECULES[*]}")"

module load anaconda3
conda activate om_p4_jax
python U_ind_benchmark.py
echo "Benchmark Python Complete"

# Loop over molecules and distance cutoffs
for molecule in "${MOLECULES[@]}"; do
    for cutoff in "${DISTANCE_CUTOFFS[@]}"; do
	# JAX vs. SAPT (Induction)
        python plot_U_ind_comp.py \
            --input_csv ${molecule}/${molecule}.csv \
            --output_png results/${molecule}_validation_cutoff_${cutoff}.png \
            --chemical_accuracy_kj 1e-6 \
            --distance_cutoff $cutoff\
	    --image ${molecule}_mol.png
	# JAX vs. SAPT (Electrostatics)
        python plot_U_ind_comp.py \
            --input_csv ${molecule}/${molecule}.csv \
            --output_png results/${molecule}_validation_cutoff_${cutoff}_es.png \
            --chemical_accuracy_kj 1e-6 \
            --distance_cutoff $cutoff\
	    --image ${molecule}_mol.png -es
    done
done
