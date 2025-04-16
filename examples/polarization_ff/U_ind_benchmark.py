import crystalatte
from crystalatte.plugins import openmm_utils, force_fields
import os
import pandas as pd


file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"


def main():
    if "MOLECULES_LIST" in os.environ:
        molecules = os.environ["MOLECULES_LIST"].split(",")
        print(molecules)
    else:
        molecules = ["pyrazole"] # ["imidazole","pyrazole", "pyrazine","acetic_acid"]
    
    for molecule in molecules:

        print(f"%%%%%%%%%%% Testing {molecule} %%%%%%%%%%")
        pkl_file       = os.path.join(molecule, molecule + ".pkl")
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

        df = pd.read_pickle(pkl_file)
        results = []
        for index, row in df.iterrows():
            qcel_mol = row["mol"]
            distance = row["Minimum Monomer Separations (A)"] 
            Uind_sapt = row["SAPT0 Induction (kJ/mol)"]
            Nmer_name = row["N-mer Name"]
            
            Uind_md, Udf, Unb = force_fields.polarization_energy_sample(
                    qcel_mol, 
                    pdb_file=pdb_file,
                    xml_file=ff_file,
                    residue_file=residue_file,
                    atom_types_map=atom_types_map,
                    update_pdb=True,
                    omm_decomp=True
            )
            
            print(Uind_md, Udf, Unb) 

            simmd = openmm_utils.setup_openmm(
                pdb_file="tmp.pdb",
                ff_file=ff_file,
                residue_file=residue_file,
                error_tol=1e-16,
                platform_name="CUDA",
            )

            Uind_omm, Udf_omm, Unb_omm = openmm_utils.U_ind_omm(simmd, decomp=True)
            
            print(Uind_omm, Udf_omm, Unb_omm)
            results.append({
                "distance": distance,
                "nmer_name": Nmer_name,
                "Uind_md": Uind_md,
                "Udf": Udf,
                "Unb": Unb,
                "Uind_sapt": Uind_sapt,
                "Uind_omm":Uind_omm,
                "Udf_omm":Udf_omm,
                "Unb_omm":Unb_omm,
            })
            print(f"(Uind_sapt, Uind_md, Uind_omm, distance) = ({Uind_sapt},{Uind_md},{Uind_omm},{distance})")
            #break
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        #break

if __name__ == "__main__":
    main()

