Herein, I'll describe this contrived "map file" concept for convenient conversion from 
separately labeled (or even unlabeled) atomic elements. To create a map file, 

## **STEP 1**: Extract the "destination" labels, i.e., here is a sapt.in file for pyrazole

molecule Dimer_0_1 {
  N            6.27981210      -4.25303372       0.19313865
  N            6.06870777      -2.28125697       1.80544317
  H            3.66871156      -5.64628941      -1.96684001
  H            0.69034521      -2.37545694      -0.42864236
  H            3.35917442       0.17650601       3.09265017
  H            7.35266782      -1.99461311       2.90680702
  C            3.77952586      -1.21508453       1.76858172
  C            2.38242998      -2.55505538       0.04607681
  C            4.01771468      -4.38410490      -0.88761863
--
...
}

The "atom types" can be labeled lazily as :

			N0
			N1
			H2
			H3
			H4
			H5
			C6
			C7
			C8

which can be turned into a quick pdb file:

ATOM      1  N0  PYR     1       6.279  -4.253   0.193
ATOM      2  N1  PYR     1       6.068  -2.281   1.805
ATOM      3  H2  PYR     1       3.668  -5.646  -1.966
ATOM      4  H3  PYR     1       0.690  -2.375  -0.428
ATOM      5  H4  PYR     1       3.359   0.176   3.092
ATOM      6  H5  PYR     1       7.352  -1.994   2.906
ATOM      7  C6  PYR     1       3.779  -1.215   1.768
ATOM      8  C7  PYR     1       2.382  -2.555   0.046
ATOM      9  C8  PYR     1       4.017  -4.384  -0.887
TER
END


## **Step 2**: Generate "reference" labels, i.e., from SAPT, OPLS, etc.

One convenient tool to create atom typed pdb files for any arbitrary molecule is 
ligpargen, i.e., here is the pdb file generated for pyrazole:

REMARK LIGPARGEN GENERATED PDB FILE
ATOM      1  C00 PYR     1       1.000   1.000   0.000
ATOM      2  C01 PYR     1      -0.374   1.000   0.000
ATOM      3  N02 PYR     1      -0.750   1.000   1.312
ATOM      4  N03 PYR     1       0.290   1.000   2.162
ATOM      5  C04 PYR     1       1.354   1.000   1.362
ATOM      6  H05 PYR     1       1.654   1.001  -0.860
ATOM      7  H06 PYR     1      -1.102   1.001  -0.799
ATOM      8  H07 PYR     1      -1.689   1.001   1.697
ATOM      9  H08 PYR     1       2.342   0.999   1.802
TER 
CONECT    1    2 
CONECT    2    3 
CONECT    3    4 
CONECT    1    5 
CONECT    1    6 
CONECT    2    7 
CONECT    3    8 
CONECT    5    9 
CONECT    4    5 
END                                   

If you have a reference pdb for another force field, that can be used as well. 
The essential ingredient is the reference atom type, 

		C00
		C01
		N02
		N03
		C04
		H05
		H06
		H07
		H08

## **Step 3**: Create csv "atom_type_map.csv" that goes from reference to destination

There are likely a number of solutions to create this map. Here, one can visualize 
and cross-reference the molecules given that they are small molecules and simple 
structures. While tedious, other solutions may be prone to edge-case errors that 
will result in poorly defined ff file. 

Here, the pyrazole types map as follows:

		C00 -> C7 
		C01 -> C6 
		N02 -> N1
		N03 -> N0
		C04 -> C8
		H05 -> H3
		H06 -> H4
		H07 -> H5
		H08 -> H2

