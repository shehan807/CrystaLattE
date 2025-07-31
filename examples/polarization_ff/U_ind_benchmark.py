import crystalatte
from crystalatte.plugins import force_fields
import os
import pandas as pd
import time 
from openff.toolkit.topology import Molecule

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
        
        df = pd.read_pickle(pkl_file)
        results = []        
        start_time = time.time()
        for index, row in df.iterrows():
            qcel_mol = row["mol"]
<<<<<<< Updated upstream
=======
            qcel_mol.to_file("pyrazine.xyz", dtype="xyz")
            #fragments = qcel_mol.fragments()
            #print(fragments)
            #off_frag = Molecule.from_qcschema(fragment.dict())
            #print(f"off_frag: {off_frag}")
            #mapped_smiles = off_frag.to_smiles(
            #    isomeric=True, 
            #    explicit_hydrogens=True, 
            #    mapped=True  # <--- critical!
            #)
            #print(f"mapped_smiles: {mapped_smiles}")
            #off_frag_canonical = Molecule.from_smiles(
            #    mapped_smiles,
            #    allow_undefined_stereo=True
            #)
            #print(f"off_frag: {off_frag}")
            #off_frag_canonical = Molecule.from_smiles(
            #    mapped_smiles,
            #    allow_undefined_stereo=True
            #)
            #print(f"off_frag_canonical: {off_frag_canonical}")

                

>>>>>>> Stashed changes
            distance = row["Minimum Monomer Separations (A)"] 
            Uind_sapt = row["SAPT0 Induction (kJ/mol)"]
            Ues_sapt = row["SAPT0 Electrostatics (kJ/mol)"]
            Nmer_name = row["N-mer Name"]
            
            Uind_md, Udf, Unb, Ues = force_fields.polarization_energy_sample(
                    qcel_mol, 
                    pdb_file=pdb_file,
                    xml_file=ff_file,
                    atom_types_map=atom_types_map,
                    omm_decomp=True
            )

            results.append({
                "distance": distance,
                "nmer_name": Nmer_name,
                "Uind_md": Uind_md,
                "Udf": Udf,
                "Unb": Unb,
                "Ues": Unb,
                "Uind_sapt": Uind_sapt,
                "Ues_sapt": Ues_sapt,
            })
            print(f"(Ues_sapt, Ues, Uind_sapt, Uind, distance) = ({Ues_sapt}, {Ues}, {Uind_sapt}, {Uind_md}, {distance})")
        end_time = time.time()
        results_df = pd.DataFrame(results)
        results_df['time_per_system'] = (end_time - start_time) / len(results)
        results_df.to_csv(output_csv, index=False)
        #break

if __name__ == "__main__":
    main()

