from crystalatte.plugins import force_fields
import qcelemental as qcel
import os

# molA causes the SCF to diverge
molA = qcel.models.Molecule.from_data(
    """
0 1
7   -2.27180868  -1.62371376   2.80737099    
7   -1.14221510  -1.54851696   0.91038297    
1   -0.01347067  -3.36505680   1.17501922    
1   -1.42119519  -3.45690432   3.50022125    
1   -2.51035542  -0.00322272   1.45849689    
1   -0.74638085  -1.18918368   0.00000000    
6   -0.76062869  -2.68560000   1.55612968    
6   -1.46529583  -2.72319840   2.71744605    
6   -2.04238609  -0.94049712   1.69915316    
--
0 1
7    0.10006198  -0.58384945  -1.47476902    
7    1.22965556  -0.65904624  -3.37175705    
1    2.35839999   1.15749360  -3.10712079    
1    0.95067546   1.24934113  -0.78191877    
1   -0.13848477  -2.20434048  -2.82364313    
1    1.62548980  -1.01837953  -4.28214002    
6    1.61124196   0.47803680  -2.72601034    
6    0.90657482   0.51563520  -1.56469396    
6    0.32948456  -1.26706609  -2.58298686    
units angstrom
"""
)
#print(molA)
#print(molA.geometry)
file_dir = os.path.dirname(os.path.realpath(__file__)) + "/" + "imidazole/"

dimer = force_fields.polarization_energy_sample(
    molA,
    pdb_file=f"{file_dir}imidazole.pdb",
    xml_file=f"{file_dir}imidazole.xml",
    residue_file=f"{file_dir}imidazole_residue.xml",
    atom_types_map=f"{file_dir}imidazole_map.csv",
)
print(dimer)
