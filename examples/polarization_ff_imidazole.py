"""
Example: compute imidazole crystal lattice energy using a Drude-polarizable
SAPT-FF via the polarization plugin's custom_function dispatch.

Run from the CrystaLattE project root:
    python examples/polarization_ff_imidazole.py
"""
from pathlib import Path

import crystalatte
from crystalatte import plugins


HERE = Path(__file__).parent
FIXTURES = HERE / "polarization_ff" / "imidazole"
SHIPPED_CIFS = HERE.parent / "crystalatte" / "data" / "cif"


def main():
    _, cle, output_data = crystalatte.main(
        cif_input=str(SHIPPED_CIFS / "Imidazole.cif"),
        cif_output="./imidazole.xyz",
        cif_a=3, cif_b=3, cif_c=3,
        bfs_thresh=1.2,
        uniq_filter="ChSEV",
        nmers_up_to=2,
        r_cut_com=8.0,
        r_cut_monomer=4.0,
        r_cut_dimer=8.0,
        cle_run_type=["custom"],
        method="drude_oscillator",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        custom_function=plugins.force_fields.polarization_energy_function,
        pdb_file=str(FIXTURES / "imidazole.pdb"),
        xml_file=str(FIXTURES / "imidazole.xml"),
        atom_types_map=str(FIXTURES / "imidazole_map.csv"),
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
