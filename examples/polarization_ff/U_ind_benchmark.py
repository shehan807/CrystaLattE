import crystalatte
from crystalatte.plugins import force_fields
import os
import pandas as pd


file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"


def main():
    molecules = ["imidazole"] # ["imidazole","pyrazole", "pyrazine","acetic_acid"]
    
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
            
            Uind_md = force_fields.polarization_energy_sample(
                    qcel_mol, 
                    pdb_file=pdb_file,
                    xml_file=ff_file,
                    residue_file=residue_file,
                    atom_types_map=atom_types_map,
            )
            results.append({
                "distance": distance,
                "nmer_name": Nmer_name,
                "Uind_md": Uind_md,
                "Uind_sapt": Uind_sapt
            })
            print(f"(Uind_sapt, Uind_md, distance) = ({Uind_sapt},{Uind_md},{distance})")
            #break
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        #break

if __name__ == "__main__":
    main()

