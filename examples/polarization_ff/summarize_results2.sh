#!/bin/bash

# Define the list of molecules and distance cutoffs
MOLECULES=("imidazole-es") # "acetic_acid" "pyrazine" "pyrazole") # You can replace this with ("imidazole" "pyrazole" "pyrazine" "acetic_acid")
DISTANCE_CUTOFFS=(2 10 12)  # Example values, modify as needed

# Export the molecules list as an environment variable
export MOLECULES_LIST="$(IFS=,; echo "${MOLECULES[*]}")"

module load anaconda3
conda activate om_p4_jax
python U_ind_benchmark2.py
echo "Benchmark Python Complete"

# Loop over molecules and distance cutoffs
for molecule in "${MOLECULES[@]}"; do
    for cutoff in "${DISTANCE_CUTOFFS[@]}"; do
	# JAX vs. SAPT
        python plot_U_ind_comp.py \
            --input_csv ${molecule}/${molecule}.csv \
            --output_png results/${molecule}_validation_cutoff_${cutoff}.png \
            --chemical_accuracy_kj 1e-6 \
            --distance_cutoff $cutoff
	# JAX vs. OpenMM 
        python plot_U_ind_comp.py \
            --input_csv ${molecule}/${molecule}.csv \
            --output_png results/${molecule}_verification_cutoff_${cutoff}.png \
            --chemical_accuracy_kj 1e-6 \
            --distance_cutoff $cutoff -v
	# JAX vs. OpenMM (DrudeForce)
        python plot_U_ind_comp.py \
            --input_csv ${molecule}/${molecule}.csv \
            --output_png results/${molecule}_verification_cutoff_${cutoff}_df.png \
            --chemical_accuracy_kj 1e-6 \
            --distance_cutoff $cutoff -df
	# JAX vs. OpenMM (NonbondedForce)
        python plot_U_ind_comp.py \
            --input_csv ${molecule}/${molecule}.csv \
            --output_png results/${molecule}_verification_cutoff_${cutoff}_nb.png \
            --chemical_accuracy_kj 1e-6 \
            --distance_cutoff $cutoff -nb
	# OpenMM vs. SAPT
        python plot_U_ind_comp.py \
            --input_csv ${molecule}/${molecule}.csv \
            --output_png results/${molecule}_omm_validation_cutoff_${cutoff}.png \
            --chemical_accuracy_kj 1e-6 \
            --distance_cutoff $cutoff -omm
    done
done
