import crystalatte
from crystalatte.plugins import force_fields
import os
import pandas as pd
import time 
from openff.toolkit.topology import Molecule
from ind20 import get_all_fit_energies
import re 

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

def extract_sapt_induction(output_file):
    """Extract Ind20,r + Exch-Ind20,r values from SAPT output file and return in kJ/mol"""
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Find Ind20,r value in kJ/mol
    ind20_match = re.search(r'Ind20,r\s+.*?\s+([-\d.]+)\s+\[kJ/mol\]', content)
    # Find Exch-Ind20,r value in kJ/mol
    exch_ind20_match = re.search(r'Exch-Ind20,r\s+.*?\s+([-\d.]+)\s+\[kJ/mol\]', content)
     
    if ind20_match and exch_ind20_match:
        ind20_value = float(ind20_match.group(1))
        exch_ind20_value = float(exch_ind20_match.group(1))
        return ind20_value, exch_ind20_value
    else:
        print(f"Ind20 and Exch-Ind20 Match not found!")
        return None, None

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
            distance = row["Minimum Monomer Separations (A)"] 
            Uind_sapt = row["SAPT0 Induction (kJ/mol)"]
            # Ues_sapt = row["SAPT0 Electrostatics (kJ/mol)"]
            Nmer_name = row["N-mer Name"]
            
            Uind_md, Udf, Unb, Ues = force_fields.polarization_energy_sample(
                    qcel_mol, 
                    pdb_file=pdb_file,
                    xml_file=ff_file,
                    atom_types_map=atom_types_map,
                    omm_decomp=True
            )
            
            output_file = os.path.join(molecule, f"{molecule}_outputs", f"{Nmer_name}.out")
            Uind20_sapt, ind_exch = extract_sapt_induction(output_file)
            fit_energies = get_all_fit_energies(qcel_mol, molecule)
            
            results.append({
                "distance": distance,
                "nmer_name": Nmer_name,
                "Uind_md": Uind_md, #fit_energies['ind20_fit'], 
                "Udf": Udf,
                "Unb": Unb,
                "Ues": Unb,
                "Uind_sapt": Uind20_sapt, #Uind_sapt,
                "ind20_fit": fit_energies['ind20_fit'],
                "dhf_fit": fit_energies['dhf_fit'],
                "Uind20": Uind20_sapt, 
                "Uexch_ind": ind_exch,
                # "Ues_sapt": Ues_sapt,
            })
            #print(f"(Ues_sapt, Ues, Uind_sapt, Uind, distance) = ({Ues_sapt}, {Ues}, {Uind_sapt}, {Uind_md}, {distance})")
            print(f"(Ues, Uind_sapt, Uind, distance) = ({Ues}, {Uind_sapt}, {Uind_md}, {distance})")
        end_time = time.time()
        results_df = pd.DataFrame(results)
        results_df['time_per_system'] = (end_time - start_time) / len(results)
        results_df.to_csv(output_csv, index=False)
        #break

if __name__ == "__main__":
    main()

