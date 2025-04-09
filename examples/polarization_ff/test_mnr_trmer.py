from crystalatte.plugins import force_fields
import qcelemental as qcel
import os

mol = qcel.models.Molecule.from_data(
    """
0 1
7   -0.24761467   2.00667417   3.39867018
7   -2.38223717   2.14877553   6.98345802
1   -4.51525502  -1.28398727   6.48336797
1   -1.8550412   -1.45755392   2.08937295
1    0.20317334   5.06895842   5.94767282
1   -3.13025548   2.82781701   8.70383249
6   -3.10333097   0.           5.76317358
6   -1.77170306  -0.07105068   3.56860369
6   -0.68116052   3.2977665    5.49289838
--
0 1
7   -0.24761467  -8.1434228    3.39867018
7   -2.38223717  -8.00132144   6.98345802
1   -4.51525502 -11.43408423   6.48336797
1   -1.8550412  -11.60765089   2.08937295
1    0.20317334  -5.08113854   5.94767282
1   -3.13025548  -7.32227995   8.70383249
6   -3.10333097 -10.15009697   5.76317358
6   -1.77170306 -10.22114764   3.56860369
6   -0.68116052  -6.85233046   5.49289838
--
0 1
7  -14.1391917  -11.25341251  13.39723417
7  -12.0045692  -11.39551386   9.81244634
1   -9.87155135  -7.96275107  10.31253638
1  -12.53176517  -7.78918441  14.7065314 
1  -14.58997971 -14.31569676  10.84823154
1  -11.25655089 -12.07455535   8.09207186
6  -11.2834754   -9.24673834  11.03273078
6  -12.6151033   -9.17568766  13.22730066
6  -13.70564584 -12.54450484  11.30300598
units bohr
"""
)
print(mol)
print(mol.geometry)
monA = mol.get_fragment(0)
print(monA)
print(monA.geometry)
file_dir = os.path.dirname(os.path.realpath(__file__)) + "/" + "imidazole/"
monomer = force_fields.polarization_energy_sample(
    mol.get_fragment(0),
    pdb_file=f"{file_dir}imidazole.pdb",
    xml_file=f"{file_dir}imidazole.xml",
    residue_file=f"{file_dir}imidazole_residue.xml",
    atom_types_map=f"{file_dir}imidazole_map.csv",
)
print(monomer)
#dimer = force_fields.polarization_energy_sample(
#    mol.get_fragment([0, 1]),
#    pdb_file=f"{file_dir}imidazole.pdb",
#    xml_file=f"{file_dir}imidazole.xml",
#    residue_file=f"{file_dir}imidazole_residue.xml",
#    atom_types_map=f"{file_dir}imidazole_map.csv",
#)
#print(dimer)
#trimer = force_fields.polarization_energy_sample(
#    mol,
#    pdb_file=f"{file_dir}imidazole.pdb",
#    xml_file=f"{file_dir}imidazole.xml",
#    residue_file=f"{file_dir}imidazole_residue.xml",
#    atom_types_map=f"{file_dir}imidazole_map.csv",
#)
#print(trimer)
