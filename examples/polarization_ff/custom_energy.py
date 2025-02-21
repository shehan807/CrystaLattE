import crystalatte
from crystalatte.plugins import force_fields
import os


file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"


def main():
    e = force_fields.openmm_inputs_polarization_energy(
        pdb_file=f"{file_dir}imidazole.pdb",
        xml_file=f"{file_dir}imidazole.xml",
        residue_file=f"{file_dir}imidazole_residue.xml",
    )
    print(e)
    # return
    # atom_types needs to be shape (N_atoms, N_molecules, 1)
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
    _, _, output_data = crystalatte.main(
        cif_input=file_dir + "./imidazole.cif",
        cif_output=file_dir + "./imidazole.xyz",
        cif_a=3,
        cif_b=3,
        cif_c=3,
        bfs_thresh=1.2,
        uniq_filter="ChSEV",
        nmers_up_to=3,
        r_cut_com=6.5,
        r_cut_monomer=3.5,
        r_cut_dimer=3.6,
        r_cut_trimer=5.7,
        r_cut_tetramer=3.7,
        r_cut_pentamer=6.1,
        cle_run_type=["custom"],
        method="my_method",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        # custom_function=force_fields.example_energy_function,
        custom_function=force_fields.polarization_energy_function,
        pdb_file=f"{file_dir}imidazole.pdb",
        xml_file=f"{file_dir}imidazole.xml",
        residue_file=f"{file_dir}imidazole_residue.xml",
        atom_types=monomer_atom_types_in_order_of_xyz,
    )
    try:
        import pandas as pd

        df = pd.DataFrame(output_data)
        print(df)
        df.to_csv("./ammonia_results.csv", index=False)
    except ImportError:
        print("Pandas not installed, printing dictionary")
        print(output_data)
    return


if __name__ == "__main__":
    main()
