# Drude-polarizable induction energies with CrystaLattE

`crystalatte.plugins.polarization_ff.polarization_energy_function` computes the
**induction (polarization) energy** of each N-mer with a Drude-oscillator force
field. It is a `custom_function` plugin for `crystalatte.main()`. For each
N-mer it relaxes the Drude displacements (SCF) and stores the non-additive
many-body induction energy, in Hartree, in `nmer["nambe"]`. Only induction is
computed: no electrostatics, no dispersion, no repulsion.

## Files

```text
examples/polarization_ff/
├── polarization_ff_imidazole.py   # driver: imidazole up to 3-body
├── pol_ff_cle.yml                 # conda env (Python 3.12, psi4, jax, jaxopt, optax, mdanalysis)
└── imidazole/
    ├── imidazole.xml              # force field (OpenMM-style XML)
    └── imidazole_map.csv          # maps XML atom names to CrystaLattE atom positions
crystalatte/data/cif/Imidazole.cif # crystal structure (the only source of coordinates)
crystalatte/tests/test_96_polarization_ff.py  # 2-body regression test using the files above
```

## Running

```bash
conda env create -f examples/polarization_ff/pol_ff_cle.yml
conda activate pol_ff_cle
pip install -e . --no-deps            # from the repo root

cd examples/polarization_ff
python polarization_ff_imidazole.py   # ~10 s; CLE = -0.002883 a.u.
```

In your own driver, pass the plugin and its two required inputs through `crystalatte.main()`:

```python
crystalatte.main(
    cif_input="my_molecule.cif",
    cle_run_type=["custom"],
    custom_function=plugins.polarization_ff.polarization_energy_function,
    xml_file="my_molecule/my_molecule.xml",
    atom_types_map="my_molecule/my_molecule_map.csv",
    ...,  # usual CrystaLattE geometry/cutoff arguments
)
```

`method`, `bsse_type` and `job_memory` are accepted but ignored. Relative paths
are resolved from the directory you run in.

## How the inputs fit together

There are three inputs, and each owns one thing:

| Input | Owns | Notes |
|---|---|---|
| CIF ([`Imidazole.cif`](../../crystalatte/data/cif/Imidazole.cif)) | **coordinates and atom order** | CrystaLattE builds monomers from it. Every monomer is reordered to match monomer 0 by bond-graph matching. |
| XML ([`imidazole.xml`](imidazole/imidazole.xml)) | **parameters** (charges, polarizabilities, Thole) | Atoms are identified by their order in the `<Residue>`. |
| map CSV ([`imidazole_map.csv`](imidazole/imidazole_map.csv)) | **which CIF atom gets which XML parameters** | One row per non-Drude residue atom, in residue order. |

Here is one atom traced through all three files: the imidazole nitrogen that carries the H.

```text
 imidazole.xml                          imidazole_map.csv            Imidazole.cif → CrystaLattE monomer
 ─────────────                          ─────────────────            ───────────────────────────────────
 <Residue>, non-Drude <Atom> #1    ──▶  row 1:  N0 , N1         ──▶  atom at position 1, element N
   <Atom name="N0" type="IM-N0"/>              ▲     ▲                (bonded to C6, C8, H5)
                                               │     └─ "To":   <element><position> in the monomer
                                               └─────── "From": the XML <Atom name>
 parameters of type IM-N0:
   core charge  0.4939   (NonbondedForce)
   Drude charge −0.8404  (DrudeForce)      ─────────────────────────▶ placed on that atom's coordinates
   α = 0.0010466 nm³, Thole a = 1
```

All counting is 0-based. **Row *i* of the map belongs to non-Drude `<Atom>` #*i* of the residue**. `From`
repeats that atom's name as a cross-check. `To` says which CrystaLattE atom it sits on.

The full imidazole map. Each row reads left to right, from the XML to the crystal:

| Row | XML `<Atom name>` = `From` | XML type | `To` = CrystaLattE atom | Bonded to (in the CIF) | Core q | Drude q | α (nm³) |
|---|---|---|---|---|---|---|---|
| 0 | N00 | IM-N00 | N0 | C7, C8 | 0.2885 | −0.8404 | 0.0010466 |
| 1 | N0  | IM-N0  | N1 | C6, C8, H5 | 0.4939 | −0.8404 | 0.0010466 |
| 2 | H2  | IM-H2  | H2 | C6 | 0.1294 | — | — |
| 3 | H21 | IM-H21 | H3 | C7 | 0.1021 | — | — |
| 4 | H1  | IM-H1  | H4 | C8 | 0.0374 | — | — |
| 5 | H0  | IM-H0  | H5 | N1 | 0.3178 | — | — |
| 6 | C2  | IM-C2  | C6 | N1, C7, H2 | 1.035 | −1.1478 | 0.00195233 |
| 7 | C21 | IM-C21 | C7 | N0, C6, H3 | 1.2032 | −1.1478 | 0.00195233 |
| 8 | C1  | IM-C1  | C8 | N0, N1, H4 | 1.5169 | −1.1478 | 0.00195233 |

The XML names (`N0`, `H2`, …) and the CrystaLattE labels (`N0`, `H2`, …) look alike
but are **independent naming schemes**. XML `N0` is CrystaLattE `N1`. They only
coincide by accident, as for `H2`. Match atoms by bonding: XML `N0` bonds an H
(`H0`), and the only N with an H in the crystal is `N1`, bonded to `H5`.

### The CIF

Example: [`crystalatte/data/cif/Imidazole.cif`](../../crystalatte/data/cif/Imidazole.cif)

This is a standard crystallographic file, read by CrystaLattE as for any other run
(`cif_input=`). It is the only source of coordinates. It also fixes the atom
order that the map's `To` labels refer to.

### The XML force field

Example: [`imidazole/imidazole.xml`](imidazole/imidazole.xml)

This is an OpenMM-style file. The plugin reads these sections:

- `<AtomTypes>/<Type>`: `name`, `class`, `element`, `mass`. **Drude particles are recognized by `class="Sh"`**. Mass is read but not used.
- `<Residues>/<Residue>`: exactly **one** residue, since only crystals of a single molecule type are supported.
  - `<Atom name=... type=...>`: the non-Drude atoms, in order, set the row order of the map. Drude atoms can be listed anywhere in the residue.
  - `<Bond from= to=>`: 0-based indices into **all** `<Atom>` lines of the residue, Drude lines included. Bonds decide which pairs within a molecule get Thole screening (pairs up to 4 bonds apart).
- `<NonbondedForce>/<Atom>`: `charge` is the **core** charge of each type. `sigma` and `epsilon` are read but not used.
- `<DrudeForce>/<Particle>`:
  - `type1` is the Drude type and `type2` its parent type.
  - `charge` is the Drude charge q_D.
  - `polarizability` is α, in **nm³**.
  - `thole` is a; the pair screening factor is a_i + a_j.
  - Atoms without a `<Particle>`, such as H here, are not polarizable.

Units follow OpenMM: nm, elementary charge, kJ/mol. The static charge of a
polarizable atom is `core + drude` charge. For example, imidazole N0 is
0.4939 − 0.8404. The molecule's total charge should be zero.

Note, current examples pull their force field parameters from
[SAPT_force_field_OpenMM](https://github.com/jmcdaniel43/SAPT_force_field_OpenMM/blob/master/sapt.xml).

### The map CSV

Example: [`imidazole/imidazole_map.csv`](imidazole/imidazole_map.csv)

```text
N00,N0
N0,N1
...
```

- No header. Two columns: `From` (the XML residue `<Atom name>`) and `To` (the CrystaLattE atom label).
- **Rows must be in the same order as the residue's non-Drude `<Atom>` lines.**
- `To` is `<element><position>`, where position is the 0-based index of the atom in a CrystaLattE monomer built from **this CIF**. That order is not the CIF's own atom-site order. Get it with the helper below; don't read it off the CIF.
- The map belongs to a specific CIF. A different CIF of the same molecule can produce a different atom order, so recheck the map.

## Adding a new molecule

1. **Get a CIF** and make sure CrystaLattE builds whole monomers from it. Use a
   supercell (`cif_a/b/c`) large enough for `r_cut_monomer`. Otherwise edge
   monomers are cut off and the plugin stops with a "composition does not match"
   error.

2. **Get Drude FF parameters** (charges, α, Thole) for the molecule and write the
   XML as described above. Producing the parameters is outside the scope of this
   example. The imidazole parameters are SAPT-FF parameters (see `<Info>` in
   `imidazole.xml`). Each polarizable atom needs one `class="Sh"` type, a
   `<NonbondedForce>` entry and a `<DrudeForce>` particle. List every bond in the
   residue.

3. **Print CrystaLattE's atom order and bonds** for monomer 0. Use them as the `To` labels:

   ```python
   import contextlib, io
   import crystalatte
   from crystalatte.crystalatte import nmers_to_qcel_mol
   from crystalatte.plugins import utils

   cif = "my_molecule.cif"
   with contextlib.redirect_stdout(io.StringIO()):
       nmers, _, _ = crystalatte.main(cif_input=cif, cif_output="tmp.xyz", cif_a=5, cif_b=5, cif_c=5,
                                      nmers_up_to=2, r_cut_com=8.0, r_cut_monomer=8.0, r_cut_dimer=8.0,
                                      cle_run_type=["test"], verbose=0)
   dimer = next(v for k, v in nmers.items() if k.startswith("2mer"))
   mono = utils._fix_topological_order(nmers_to_qcel_mol(dimer)).get_fragment(0)
   labels = [f"{s}{i}" for i, s in enumerate(mono.symbols)]
   bonds = sorted(tuple(sorted(map(int, e))) for e in utils._create_graph(mono).edges)
   print("To labels:", labels)
   print("bonds:    ", [(labels[i], labels[j]) for i, j in bonds])
   ```

   For `Imidazole.cif` this prints:

   ```text
   To labels: ['N0', 'N1', 'H2', 'H3', 'H4', 'H5', 'C6', 'C7', 'C8']
   bonds:     [('N0', 'C7'), ('N0', 'C8'), ('N1', 'H5'), ('N1', 'C6'), ('N1', 'C8'), ('H2', 'C6'), ('H3', 'C7'), ('H4', 'C8'), ('C6', 'C7')]
   ```

   Here `N1` carries the H, so it gets the N–H type (`N0` in the XML).

4. **Write the map**: one row per non-Drude residue atom, in residue order. Match
   each XML atom to the CrystaLattE label with the same bonding: same element,
   same neighbors.

5. **Check the map against the XML bonds.** Run this right after step 3, so that
   `labels` and `bonds` are already defined. The plugin's own input checks only
   compare element counts. A map that swaps two atoms of the same element (for
   example the two imidazole N's) passes those checks and silently gives wrong
   energies. This check catches it.

   ```python
   import pandas as pd

   xml, csv = "my_molecule/my_molecule.xml", "my_molecule/my_molecule_map.csv"
   xmlmd = utils.XmlMD(); xmlmd.parse_xml(xml)
   res = next(iter(xmlmd.residues.values()))
   core = [(n, t) for n, t in res if xmlmd.atom_types[t]["class"] != "Sh"]
   m = pd.read_csv(csv, names=["From", "To"])
   assert list(m["From"]) == [n for n, _ in core], "map rows must follow the residue's non-Drude <Atom> order"
   name2to = dict(zip(m["From"], m["To"]))
   type2name = {t: n for n, t in res}
   xml_bonds = {frozenset((name2to[type2name[a]], name2to[type2name[b]])) for a, b in xmlmd.bonds}
   cif_bonds = {frozenset((labels[i], labels[j])) for i, j in bonds}
   print("map OK" if xml_bonds == cif_bonds else
         f"MISMATCH\n  only in XML: {xml_bonds - cif_bonds}\n  only in CIF: {cif_bonds - xml_bonds}")
   ```

   For a symmetric molecule (e.g. pyrazine), several maps can pass. That is fine
   as long as symmetry-equivalent atoms have identical parameters.

6. **Run a 2-body calculation first.** Copy `polarization_ff_imidazole.py`,
   change the paths, and set `nmers_up_to=2`. Check that the dimer `nambe` values
   are negative and of sensible size, and compare a few against SAPT induction if
   you have it. Then go to 3-body. Adding a regression test like
   `test_96_polarization_ff.py` keeps the new molecule from breaking later.

## Common errors

| Error | Cause / fix |
|---|---|
| `requires \`xml_file=...\`` | Pass `xml_file` / `atom_types_map` as keyword arguments to `crystalatte.main()`. |
| `` `xml_file` not found `` | The path is relative to the directory you run in, which the message prints. |
| `Monomer i composition ... does not match` | Supercell too small for `r_cut_monomer` (increase `cif_a/b/c`), or the map is missing atoms. |
| `Bond count mismatch` / `not isomorphic` | Monomers in the CIF are not all the same molecule, or `bfs_thresh` joins or splits molecules. |
| `Multiple residues found` | The XML must contain exactly one `<Residue>`. |
| `IndexError` in `get_Rij_Dij` | A `To` label is missing or has the wrong element for its position. Rerun steps 3–5. |
