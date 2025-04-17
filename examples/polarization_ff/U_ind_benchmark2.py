import crystalatte
from crystalatte.plugins import openmm_utils, force_fields
import os
import pandas as pd
import time 
import numpy as np 

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"


def main():
    optimizers = ["BFGS", "LBFGS", "NonlinearCG"] #"GradientDescent", "NonlinearCG", "PolyakSGD"]
    tolerances = [1e-3, 1e-8, 1e-16]
    objective_functions = ["Uind"] #, "Udf"]

    if "MOLECULES_LIST" in os.environ:
        molecules = os.environ["MOLECULES_LIST"].split(",")
        print(molecules)
    else:
        molecules = ["pyrazole"] # ["imidazole","pyrazole", "pyrazine","acetic_acid"]
    
    for molecule in molecules:
        for optimizer in optimizers:
            for tolerance in tolerances:
                for objective_function in objective_functions:

                    print(f"%%%%%%%%%%% Testing {molecule} %%%%%%%%%%")
                    pkl_file       = os.path.join(molecule, molecule + ".pkl")
                    ref_pkl_file   = os.path.join(molecule, molecule + "-ref.pkl")
                    pdb_file       = os.path.join(molecule, molecule + ".pdb")
                    ff_file        = os.path.join(molecule, molecule + ".xml")
                    residue_file   = os.path.join(molecule, molecule + "_residue.xml")
                    atom_types_map = os.path.join(molecule, molecule + "_map.csv")
                    output_csv     = os.path.join(molecule, molecule + ".csv") 
                    
                    # U_ind based on pdb-fed positions (deprecated, left for demonstration)
                    # Uind_dummy = force_fields.openmm_inputs_polarization_energy(
                    #         pdb_file=pdb_file,
                    #         xml_file=ff_file,
                    #         residue_file=residue_file,
                    # )
                    # print(f"Uind_openmm for {pdb_file} = {Uind_dummy}.")
                    # print(f"openmm_inputs_polarization_energy works!")

                    df     = pd.read_pickle(pkl_file)
                    print(df.columns)
                    results = []
                    unconverged = 0
                    total = 0
                    start = time.time()
                    conv_omm = []
                    conv_jax = []
                    for index, row in df.iterrows():
                        total += 1
                        qcel_mol = row["mol"]
                        distance = row["Minimum Monomer Separations (A)"] 
                        Nmer_name = row["N-mer Name"]
                        Ues_sapt = row["SAPT0 Electrostatics (kJ/mol)"] 
                        Ues_md, Udf, Unb = force_fields.polarization_energy_sample(
                                qcel_mol, 
                                pdb_file=pdb_file,
                                xml_file=ff_file,
                                residue_file=residue_file,
                                atom_types_map=atom_types_map,
                                update_pdb=True,
                                omm_decomp=True,
                                optimizer=optimizer, 
                                tolerance=tolerance, 
                                objective_function=objective_function,
                        )
                        simmd = openmm_utils.setup_openmm(
                            pdb_file="tmp.pdb",
                            ff_file=ff_file,
                            residue_file=residue_file,
                            error_tol=1e-16,
                            platform_name="CPU",
                        )

                        Ues_omm, Udf_omm, Unb_omm = openmm_utils.U_ind_omm(simmd, decomp=True)

                        if abs(Uind_md-Uind_omm) > 100:
                            unconverged += 1
                        else:
                            conv_omm.append(Uind_omm)
                            conv_jax.append(Uind_md)
                        
                        results.append({
                            "distance": distance,
                            "nmer_name": Nmer_name,
                            "Ues_sapt": Ues_sapt,
                            "Ues_md": Uind_md,
                            "Udf": Udf,
                            "Unb": Unb,
                            "Ues_omm":Uind_omm,
                            "Udf_omm":Udf_omm,
                            "Unb_omm":Unb_omm,
                        })

                        print(f"(Ues_md, Ues_omm, distance) = ({Ues_md},{Ues_omm},{distance})")
                        #if total > 50:
                        #    break
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(output_csv, index=False)
                    end = time.time()
                    RMSE = np.sqrt(np.mean((np.array(conv_omm) - np.array(conv_jax)) ** 2))
                    print(f"{optimizer},tol={tolerance}, obj={objective_function}:\n{total-unconverged} / {total} Converged systems!!!\nTime: {(end-start)/total} seconds/system.\nRMSE={RMSE}.")
                    #break

if __name__ == "__main__":
    main()

