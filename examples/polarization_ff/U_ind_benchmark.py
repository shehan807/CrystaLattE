import crystalatte
from crystalatte.plugins import force_fields
import os
import pandas as pd


file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"


def main():
    molecules = ["imidazole","pyrazole","acetic_acid"]
    monomer_atom_types_in_order_of_xyz = [
        "N00",
        "N0",
        "H2",
        "H21",
        "H1",
        "H0",
        "C2",
        "C21",
        "C1",
    ]
    
    for molecule in molecules:
        print(f"%%%%%%%%%%% Testing {molecule} %%%%%%%%%%")
        pkl_file     = os.path.join(file_dir, molecule + ".pkl")
        pdb_file     = os.path.join(file_dir, molecule + ".pdb")
        ff_file      = os.path.join(file_dir, molecule + ".xml")
        residue_file = os.path.join(file_dir, molecule + "_residue.xml")
        
        # U_ind based on pdb-fed positions
        Uind_dummy = force_fields.openmm_inputs_polarization_energy(
                pdb_file=pdb_file,
                xml_file=ff_file,
                residue_file=residue_file,
        )
        print(f"Uind_openmm for {pdb_file} = {Uind_dummy}.")
        print(f"openmm_inputs_polarization_energy works!")

        df = pd.read_pickle(pkl_file)
        results = []
        for index, row in df.iterrows():
            qcel_mol = row["mol"]
            distance = row["Minimum Monomer Separations (A)"] 
            Uind_sapt = row["SAPT0 Induction (kJ/mol)"]
            Nmer_name = row["N-mer Name"]
            print(qcel_mol, distance, Uind_sapt, Nmer_name)
            Uind_md = force_fields.polarization_energy_sample(
                    qcel_mol, 
                    pdb_file=pdb_file,
                    xml_file=ff_file,
                    residue_file=residue_file,
                    atom_types=monomer_atom_types_in_order_of_xyz,
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
        results_df.to_csv(molecule + ".csv", index=False)
        #break

    #e = force_fields.openmm_inputs_polarization_energy(
    #    pdb_file=f"{file_dir}imidazole.pdb",
    #    xml_file=f"{file_dir}imidazole.xml",
    #    residue_file=f"{file_dir}imidazole_residue.xml",
    #)
    #print(e)
    # return
    # atom_types needs to be shape (N_atoms, N_molecules, 1)
    #monomer_atom_types_in_order_of_xyz = [
    #    "N00",
    #    "N0",
    #    "H2",
    #    "H21",
    #    "H1",
    #    "H0",
    #    "C2",
    #    "C21",
    #    "C1",
    #]
    #_, _, output_data = crystalatte.main(
    #    cif_input=file_dir + "./imidazole.cif",
    #    cif_output=file_dir + "./imidazole.xyz",
    #    cif_a=3,
    #    cif_b=3,
    #    cif_c=3,
    #    bfs_thresh=1.2,
    #    uniq_filter="ChSEV",
    #    nmers_up_to=3,
    #    r_cut_com=6.5,
    #    r_cut_monomer=3.5,
    #    r_cut_dimer=3.6,
    #    r_cut_trimer=5.7,
    #    r_cut_tetramer=3.7,
    #    r_cut_pentamer=6.1,
    #    cle_run_type=["custom"],
    #    method="my_method",
    #    bsse_type=None,
    #    job_memory=None,
    #    verbose=2,
    #    # custom_function=force_fields.example_energy_function,
    #    custom_function=force_fields.polarization_energy_function,
    #    pdb_file=f"{file_dir}imidazole.pdb",
    #    xml_file=f"{file_dir}imidazole.xml",
    #    residue_file=f"{file_dir}imidazole_residue.xml",
    #    atom_types=monomer_atom_types_in_order_of_xyz,
    #)
    #try:
    #    import pandas as pd

    #    df = pd.DataFrame(output_data)
    #    print(df)
    #    df.to_csv("./ammonia_results.csv", index=False)
    #except ImportError:
    #    print("Pandas not installed, printing dictionary")
    #    print(output_data)
    #return


if __name__ == "__main__":
    main()

