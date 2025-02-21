import crystalatte
from crystalatte.plugins import force_fields


def main():
    e = force_fields.openmm_inputs_polarization_energy(
        pdb_file="/home/amwalla3/gits/U_pol/benchmarks/OpenMM/imidazole/imidazole.pdb",
        xml_file="/home/amwalla3/gits/U_pol/benchmarks/OpenMM/imidazole/imidazole.xml",
        residue_file="/home/amwalla3/gits/U_pol/benchmarks/OpenMM/imidazole/imidazole_residue.xml",
    )
    print(e)
    

    return
    _, _, output_data = crystalatte.main(
        cif_input="../Tests/Ammonia/Ammonia.cif",
        cif_output="./ammonia.xyz",
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
        custom_function=force_fields.example_energy_function,
        example_extra_arg=0.05,
    )
    try:
        import pandas as pd
        df = pd.DataFrame(output_data)
        print(df)
        df.to_csv("./ammonia_results.csv", index=False)
    except (ImportError):
        print("Pandas not installed, printing dictionary")
        print(output_data)
    return


if __name__ == "__main__":
    main()
