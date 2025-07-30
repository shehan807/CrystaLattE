import crystalatte
from crystalatte.plugins import force_fields
import os
import pandas as pd
import time 
import re
import matplotlib.pyplot as plt
import numpy as np
from fortran_induction import get_all_fit_energies

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"


def extract_sapt_induction(output_file):
    """Extract Ind20,r + Exch-Ind20,r values from SAPT output file and return in kJ/mol"""
    try:
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
            return None
    except:
        return None


def main():
    if "MOLECULES_LIST" in os.environ:
        molecules = os.environ["MOLECULES_LIST"].split(",")
        print(molecules)
    else:
        molecules = ["pyrazole"] # ["imidazole","pyrazole", "pyrazine","acetic_acid"]
    
    # Create figure for combined plot
    plt.figure(figsize=(10, 8))
    
    # Define colors for each molecule
    colors = {'imidazole': 'blue', 'pyrazole': 'red', 'pyrazine': 'green', 'acetic_acid': 'purple'}
    
    # Collect all y-values for determining range
    all_y_values = []
    
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
            #qcel_mol.to_file(f"{molecule}_{index}.xyz",dtype='xyz')
            #qcel_mol.to_file(f"{molecule}_{index}.in",dtype='psi4')
            distance = row["Minimum Monomer Separations (A)"] 
            Uind_sapt = row["SAPT0 Induction (kJ/mol)"]
            Ues_sapt = row["SAPT0 Electrostatics (kJ/mol)"]
            Uexch_sapt = row["SAPT0 Exchange (kJ/mol)"]
            print(f"SAPT0 Exchange: {Uexch_sapt}")
            Nmer_name = row["N-mer Name"]
            
            # Look up the output file and extract Ind20,r + Exch-Ind20,r
            output_file = os.path.join(molecule, f"{molecule}_outputs", f"{Nmer_name}.out")
            Uind20_sapt, ind_exch = extract_sapt_induction(output_file)
            print(f"ind_exch = {ind_exch} @ {distance} Angstroms")
            if ind_exch == 0.0:
                print(f"ind_exch = {ind_exch} @ {distance} Angstroms")
                #break
            sapt_ind_exch_ind = Uind20_sapt + ind_exch
            Udhf_sapt = Uind_sapt - sapt_ind_exch_ind
            Uind_md, Udf, Unb, Ues = force_fields.polarization_energy_sample(
                    qcel_mol, 
                    pdb_file=pdb_file,
                    xml_file=ff_file,
                    atom_types_map=atom_types_map,
                    omm_decomp=True
            )
            
            # Calculate all energy components using Fortran fitting code
            fit_energies = get_all_fit_energies(qcel_mol, molecule)

            #print(row)
            results.append({
                "distance": distance,
                "nmer_name": Nmer_name,
                "Uind_md": Uind_md,
                "Udf": Udf,
                "Unb": Unb,
                "Ues": Unb,
                "Uind20": Uind20_sapt, 
                "Uexch_ind": ind_exch, 
                "Udhf_sapt": Udhf_sapt,
                "Uind_sapt": Uind_sapt,
                "Ues_sapt": Ues_sapt,
                "sapt_ind_exch_ind": sapt_ind_exch_ind,
                "ind20_fit": fit_energies['ind20_fit'],
                "elec_fit": fit_energies['elec_fit'],
                "dhf_fit": fit_energies['dhf_fit'],
            })
            print(f"{index}: (Ues_sapt, Ues, Uind_sapt, Uind, distance, sapt_ind_exch_ind) = ({Ues_sapt}, {Ues}, {Uind_sapt}, {Uind_md}, {distance}, {sapt_ind_exch_ind})")
            #break
        end_time = time.time()
        results_df = pd.DataFrame(results)
        results_df['time_per_system'] = (end_time - start_time) / len(results)
        results_df.to_csv(output_csv, index=False)
        
        # Sort by distance for plotting
        results_df = results_df.sort_values('distance')
        
        # Filter for distances >= 12 Angstroms
        results_df_filtered = results_df[results_df['distance'] >= 0]
        
        # Collect y-values for range determination
        if len(results_df_filtered) > 0:
            all_y_values.extend(results_df_filtered['Udhf_sapt'].tolist())
            all_y_values.extend(results_df_filtered['sapt_ind_exch_ind'].tolist())
        
        # Get color for this molecule
        color = colors.get(molecule, 'black')
        
        # Plot Udhf_sapt (dashed line) and sapt_ind_exch_ind (solid line)
        #plt.plot(results_df_filtered['distance'], results_df_filtered['Udhf_sapt'], 
        #         linestyle='--', color=color, label=f'{molecule} Udhf_sapt', linewidth=2)
        plt.semilogy(results_df_filtered['distance'], results_df_filtered['Uexch_ind'], 
                 linestyle='-', color=color, label=f'{molecule} Uind-exch', linewidth=2)
        
        #break
    
    # Configure plot
    plt.xlabel('Distance (Å)', fontsize=14)
    plt.ylabel('U_exch-ind Energy (kJ/mol)', fontsize=14)
    #plt.title('Udhf (dashed) and Uind2+Uind-exch (solid) vs Distance', fontsize=16)
    
    # Set y-axis limits based on collected values
    #if all_y_values:
    #    y_min = max(all_y_values)
    #    y_max = min(all_y_values)
    #    # Add 5% padding
    #    y_padding = (y_max - y_min) * 0.05
    #    plt.ylim(y_min - y_padding, y_max + y_padding)
    
    plt.legend(fontsize=10)
    
    # Add refined grid lines for semilogy plot
    plt.grid(True, which='major', linestyle='-', alpha=0.4, linewidth=1.2)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.8)
    
    # Get current axes and enable minor ticks
    ax = plt.gca()
    ax.minorticks_on()
    
    # For y-axis (log scale), add more minor ticks
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=10))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('combined_Udhf_sapt_ind_exch_ind_plot.png', dpi=300)
    #plt.show()

if __name__ == "__main__":
    main()

