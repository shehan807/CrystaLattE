import crystalatte
from crystalatte import plugins


def main():
    """
cif_input       = cif/imidazole.cif  
cif_output      = mp2_q5/imidazole.xyz
nmers_up_to     = 3 				
r_cut_com       = 1000				
r_cut_monomer   = 120
r_cut_dimer     = 20
r_cut_trimer    = 20
cle_run_type    = psithon
psi4_method     = mp2/cc-pv[q5]z
psi4_bsse       = cp		
psi4_memory     = 32 GB	
verbose         = 2
    """
    _, _, output_data = crystalatte.main(
        cif_input="../Cifs/imidazole.cif",
        cif_output="./imidazole.xyz",
        nmers_up_to=3,
        # r_cut_com=1000,
        # r_cut_monomer=120,
        # r_cut_dimer=20,
        # r_cut_trimer=20,
        r_cut_com=10,
        r_cut_monomer=12,
        r_cut_dimer=2,
        r_cut_trimer=2,
        r_cut_tetramer=0,
        r_cut_pentamer=0,
        cle_run_type=["custom"],
        method="my_method",
        bsse_type=None,
        job_memory=None,
        verbose=2,
        custom_function=plugins.force_fields.U_ind,
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
