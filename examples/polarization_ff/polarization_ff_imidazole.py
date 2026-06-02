"""
Example: compute imidazole crystal lattice energy using a Drude-polarizable
SAPT-FF via the polarization plugin's custom_function dispatch.

Run from this directory (examples/polarization_ff/):
    python polarization_ff_imidazole.py
"""
import crystalatte
from crystalatte import plugins


def main():
    _, cle, output_data = crystalatte.main(
        cif_input="../../crystalatte/data/cif/Imidazole.cif",
        cif_output="./imidazole.xyz",
        cif_a=3, cif_b=3, cif_c=3,
        bfs_thresh=1.2,
        uniq_filter="ChSEV",
        nmers_up_to=3,
        r_cut_com=8.0,
        r_cut_monomer=4.0,
        r_cut_dimer=8.0,
        r_cut_trimer=8.0,
        cle_run_type=["custom"],
        method="drude_oscillator",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        custom_function=plugins.polarization_ff.polarization_energy_function,
        pdb_file="imidazole/imidazole.pdb",
        xml_file="imidazole/imidazole.xml",
        atom_types_map="imidazole/imidazole_map.csv",
    )

    print(f"\nImidazole crystal lattice energy (Drude polarizable FF): {cle:.6f} a.u.\n")

    try:
        import pandas as pd
        df = pd.DataFrame(output_data)
        print(df)
        df.to_csv("./imidazole_results.csv", index=False)
    except ImportError:
        print(output_data)


if __name__ == "__main__":
    main()
