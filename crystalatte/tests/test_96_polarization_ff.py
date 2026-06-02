"""
Unit and regression test for the crystalatte package.
"""

# Import package, test suite, and other packages as needed
from qcelemental.testing import compare, compare_values
import crystalatte
import pytest
import subprocess

polarization_ff = pytest.importorskip("crystalatte.plugins.polarization_ff")


def test_polarization_ff_imidazole():
    """Checks the Drude-polarizable force-field induction plugin through
    crystalatte.main()'s custom_function dispatch, using the same driver setup
    as examples/polarization_ff/ but truncated at 2-body for a fast regression
    test (the example itself runs up to 3-body)."""

    nmers, cle, output_data = crystalatte.main(
        cif_input="crystalatte/data/cif/Imidazole.cif",
        cif_output="examples/polarization_ff/_test_96_imidazole.xyz",
        cif_a=3,
        cif_b=3,
        cif_c=3,
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
        custom_function=polarization_ff.polarization_energy_function,
        pdb_file="examples/polarization_ff/imidazole/imidazole.pdb",
        xml_file="examples/polarization_ff/imidazole/imidazole.xml",
        atom_types_map="examples/polarization_ff/imidazole/imidazole_map.csv",
    )

    # Test the number of N-mers.
    number_di = len([k for k in nmers.keys() if k.startswith("2mer-")])
    assert compare(5, number_di, "Number of Dimers: ")

    # Test replicas for each N-mer.
    assert compare_values(1, nmers["2mer-0+1"]["replicas"], "2mer-0+1 Replicas: ")
    assert compare_values(1, nmers["2mer-0+2"]["replicas"], "2mer-0+2 Replicas: ")
    assert compare_values(1, nmers["2mer-0+3"]["replicas"], "2mer-0+3 Replicas: ")
    assert compare_values(2, nmers["2mer-0+4"]["replicas"], "2mer-0+4 Replicas: ")
    assert compare_values(1, nmers["2mer-0+6"]["replicas"], "2mer-0+6 Replicas: ")

    # Test the non-additive many-body energy (induction, in Hartree) for each N-mer.
    assert compare_values(-0.0003300230514625049, nmers["2mer-0+1"]["nambe"], atol=1.e-7)
    assert compare_values(-0.003133140267716733,  nmers["2mer-0+2"]["nambe"], atol=1.e-7)
    assert compare_values(-0.00011670026447170676, nmers["2mer-0+3"]["nambe"], atol=1.e-7)
    assert compare_values(-0.00037073024999249155, nmers["2mer-0+4"]["nambe"], atol=1.e-7)
    assert compare_values(-0.0005407274379452894,  nmers["2mer-0+6"]["nambe"], atol=1.e-7)

    # Test the crystal lattice energy.
    assert compare_values(-0.0024310257607906083, cle, atol=1.e-7)

    # Clean-up generated test files.
    subprocess.call(["rm", "-r", "examples/polarization_ff/_test_96_imidazole"])
    subprocess.call(["rm", "examples/polarization_ff/_test_96_imidazole.xyz"])
