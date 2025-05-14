import os
import jax.numpy as jnp
from optax import safe_norm
import time
import qcelemental as qcel
from qcelemental import constants
import numpy as np
import pandas as pd
from pathlib import Path
from MDAnalysis import Universe, AtomGroup
import xml.etree.ElementTree as ET
from copy import deepcopy
import tempfile
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

def visualize_isomorphism(G1, G2, mapping=None):
    """
    Visualize two isomorphic graphs side by side with the mapping highlighted.
    Args:
        G1, G2: The two graphs
        mapping: Dict mapping nodes from G1 to G2 (optional)
    """
    if mapping is None:
        GM = nx.isomorphism.GraphMatcher(G1, G2, 
                node_match=lambda n1, n2: n1['symbol'] == n2['symbol'])
        if GM.is_isomorphic():
            mapping = GM.mapping
        else:
            print("Graphs are not isomorphic!")
            return
    
    pos1 = nx.spring_layout(G1, seed=42)
    pos2 = nx.spring_layout(G2, seed=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    nx.draw_networkx_edges(G1, pos1, ax=ax1)
    nx.draw_networkx_labels(G1, pos1, 
                           labels={n: f"{G1.nodes[n]['symbol']}{n}" for n in G1.nodes()},
                           font_size=12, ax=ax1)
    
    nx.draw_networkx_edges(G2, pos2, ax=ax2)
    nx.draw_networkx_labels(G2, pos2, 
                           labels={n: f"{G2.nodes[n]['symbol']}{n}" for n in G2.nodes()},
                           font_size=12, ax=ax2)
    
    # Color-code nodes by their mapping
    cmap = plt.cm.rainbow
    colors = cmap(np.linspace(0, 1, len(G1.nodes())))
    
    # Draw nodes with mapped colors
    for i, node1 in enumerate(G1.nodes()):
        nx.draw_networkx_nodes(G1, pos1, nodelist=[node1], 
                              node_color=[colors[i]], ax=ax1)
        node2 = mapping[node1]  # Mapped node in G2
        nx.draw_networkx_nodes(G2, pos2, nodelist=[node2], 
                              node_color=[colors[i]], ax=ax2)
    
    ax1.set_title("Graph 1")
    ax2.set_title("Graph 2")
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

def _create_graph(fragment_mol):
    """Create a NetworkX graph representing the molecular structure."""
    G = nx.Graph()
    
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=True) as tmp:
        fragment_mol.to_file(tmp.name, dtype='xyz')
        u = Universe(tmp.name, format='xyz', to_guess=['bonds'])
        
    # Add nodes for each atom with symbol as attribute
    for i, symbol in enumerate(fragment_mol.symbols):
        G.add_node(i, symbol=symbol)
    
    # Add edges for bonds
    for bond in u.bonds:
        i, j = bond.indices
        G.add_edge(i, j)
    return G

def _check_isomorphism(G1, G2):
    """
    Check if two molecular graphs are isomorphic,
    considering both structure and atom symbols.
    
    Returns:
        bool: True if isomorphic, False otherwise
        dict: Node mapping if isomorphic, None otherwise
    """
    def node_match(n1, n2):
        return n1['symbol'] == n2['symbol']
    result = nx.is_isomorphic(G1, G2, node_match=node_match)
    if result:
        GM = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match)
        mapping = GM.mapping if GM.is_isomorphic() else None
        return True, mapping
    return False, None

def _check_match(G1, G2):
    """
    Check if two graphs are exactly equal - same nodes with same indices,
    same edges, and same node attributes.
    
    Returns:
        bool: True if exactly equal, False otherwise
    """
    if set(G1.nodes()) != set(G2.nodes()):
        return False
    # e.g., for pyrazine, this fails
    if set(G1.edges()) != set(G2.edges()):
        return False
    # Check if node attributes match
    for node in G1.nodes():
        if G1.nodes[node] != G2.nodes[node]:
            return False
    return True

def _fix_topological_order(qcel_mol):
    """
    Fix topological ordering issues in a QCElemental molecule object with multiple fragments.
    
    This function ensures that all fragments in the molecule have the same topological ordering
    as the first (reference) fragment. It does this by:
    
    1. Creating a list of all fragments within the QCElemental molecule 
    2. Using the first fragment as the reference, obtains bonded information via MDAnalysis
    3. Uses NetworkX's GraphMatcher to map any subsequent (isomorphic) fragments to the reference 
 
    Parameters
    ----------
    qcel_mol : qcelemental.models.Molecule
        QCElemental molecule object with one or more fragments
        
    Returns
    -------
    qcel_mol : qcelemental.models.Molecule
        QCElemental molecule with consistent topological ordering across all fragments
    """
    if not hasattr(qcel_mol, 'fragments') or len(qcel_mol.fragments) <= 1:
        print('no fragments found')
        return qcel_mol
    
    # Create the reference graph from the first fragment
    reference_mol = qcel_mol.get_fragment(0)
    reference_graph = _create_graph(reference_mol)
    ref_symbols = reference_mol.symbols
   	 
    # Track if any reordering was needed
    topology_fixed = False
    
    reordered_fragments = []
    reordered_fragments.append(reference_mol)  # First fragment is the reference
    
    for i in range(1, len(qcel_mol.fragments)):
        current_mol = qcel_mol.get_fragment(i)
        current_graph = _create_graph(current_mol)
        
        # First check if the number of bonds is different - this is a more serious issue
        if len(reference_graph.edges) != len(current_graph.edges):
            error_msg = f"Bond count mismatch for fragment {i}.\n"
            error_msg += f"Reference has {len(reference_graph.edges)} bonds, current fragment has {len(current_edges.edges)} bonds.\n"
            error_msg += "Fragments must have the same number of bonds for topological ordering."
            raise ValueError(error_msg)
        
        isomorphic, mapping = _check_isomorphism(reference_graph, current_graph)
        match               = _check_match(reference_graph, current_graph)
        mismatch            = not match
  
        if mismatch:
            if not isomorphic:
                error_msg = f"Fragment {i} has incompatible bond structure with reference fragment."
                error_msg += f"Graphs are not isomorphic, so reordering is not possible."
                raise ValueError(error_msg)
                
            reordered_geometry = np.zeros_like(current_mol.geometry)
            reordered_symbols = []
            
            for ref_idx in range(len(reference_mol.symbols)):
                curr_idx = mapping[ref_idx]  # Get corresponding index in current fragment
                reordered_geometry[ref_idx] = current_mol.geometry[curr_idx]
                reordered_symbols.append(current_mol.symbols[curr_idx])
            
            # Create a new molecule with reordered geometry
            reordered_mol = qcel.models.Molecule(
                symbols=reordered_symbols,
                geometry=reordered_geometry,
                name=f"reordered_fragment_{i}",
            )
            
            # Verify the reordering fixed the topology
            reordered_graph = _create_graph(reordered_mol)
            if not _check_match(reference_graph, reordered_graph):
                print("Warning: Reordering did not fix topology mismatch.")
                # visualize graphs to debug
                # visualize_isomorphism(reference_graph, reordered_graph)
            else:
                print(f"Successfully reordered fragment {i} to match reference topology.")
                topology_fixed = True
                print(f"REFERENCE:\nG.N: {reference_graph.nodes}\nG.E: {reference_graph.edges}") 
                print(f"NEW FRAGMENT {i}:\nG.N: {reordered_graph.nodes}\nG.E: {reordered_graph.edges}") 
            
            reordered_fragments.append(reordered_mol)
        else:
            # No mismatch, so use the original fragment
            reordered_fragments.append(current_mol)
            continue
            
    # If no fragments were reordered, return the original molecule
    if not topology_fixed:
        print("No topology reordering was needed.")
        return qcel_mol
    
    # Create a new combined molecule with all reordered fragments
    combined_geoms = np.vstack([frag.geometry for frag in reordered_fragments])
    combined_symbols = []
    for frag in reordered_fragments:
        combined_symbols.extend(frag.symbols)
        
    # Create fragments list for the combined molecule
    fragments = []
    atom_counter = 0
    for frag in reordered_fragments:
        n_atoms = len(frag.symbols)
        fragments.append(list(range(atom_counter, atom_counter + n_atoms)))
        atom_counter += n_atoms
        
    # Create the fixed molecule with reordered fragments
    fixed_mol = qcel.models.Molecule(
        symbols=combined_symbols,
        geometry=combined_geoms,
        name=qcel_mol.name if hasattr(qcel_mol, 'name') else "fixed_topology_molecule",
        fragments=fragments
    )
    
    print("Topology was fixed for one or more fragments.")
    return fixed_mol

from openmm.vec3 import Vec3
from openmm.app import (
    Simulation,
    Topology,
    ForceField,
    PDBFile,
    Modeller,
)
from openmm import (
    DrudeForce,
    NonbondedForce,
    DrudeSCFIntegrator,
    Platform,
)
from openmm.unit import (
    elementary_charge,
    picoseconds,
    nanometer,
    kilojoules_per_mole,
)

M_PI = 3.14159265358979323846
E_CHARGE = 1.602176634e-19
AVOGADRO = 6.02214076e23
EPSILON0 = 1e-6 * 8.8541878128e-12 / (E_CHARGE * E_CHARGE * AVOGADRO)
ONE_4PI_EPS0 = 1 / (4 * M_PI * EPSILON0)
import xml.etree.ElementTree as ET

import xml.etree.ElementTree as ET

class XmlMD:
    """
    A single container for all parsed data from an OpenMM-style XML file,
    plus additional attributes such as a QCElemental Molecule object and
    a custom mapping from QCEngine/QCElemental to the parsed atom types.
    """

    def __init__(self, qcel_mol=None, atom_types_map=None):
        """
        Initialize the XmlMD object.

        Parameters
        ----------
        qcel_mol : Molecule-like object, optional
            A QCElemental or QCEngine "Molecule" object (or similar).
        atom_types_map : str, optional
            A user-defined .csv file mapping from (some QCEngine label) -> 
            (XML atom type).
        """
        # The parsed data from the XML:
        self.atom_types   = {}
        self.residues     = {}
        self.bonds        = []
        self.nonbonded_params  = {}
        self.drude_params = {}
        self.core_atom_types = []

        # Additional attributes:
        self.qcel_mol         = qcel_mol
        self.atom_types_map   = pd.read_csv(atom_types_map, names=["From", "To"])

    def _findExclusions(self, bonds, atoms, drudes, maxSeparation=4):
        """
        Identify pairs of atoms in the same molecule separated by no more than
        `maxSeparation` bonds. Adapted from OpenMM forcefield.py's _findExclusions().
        - `bonds` and `atoms` contain atom type name's, i.e., str type
        """

        # map each atom type string -> integer index
        type2idx = {t: i for i, t in enumerate(atoms)}
        numAtoms = len(atoms)

        # convert your bond list ("atomTypeA", "atomTypeB") into (idx, idx)
        bondIndices = []
        for a, b in bonds:
            i = type2idx[a]
            j = type2idx[b]
            bondIndices.append((i, j))

        # verbatim logic from OpenMM's forcefield.py
        bondedTo = [set() for i in range(numAtoms)]
        for i, j in bondIndices:
            bondedTo[i].add(j)
            bondedTo[j].add(i)
        # Identify all neighbors of each atom with each separation.
        bondedWithSeparation = [bondedTo]
        for i in range(maxSeparation-1):
            lastBonds = bondedWithSeparation[-1]
            newBonds = deepcopy(lastBonds)
            for atom in range(numAtoms):
                for a1 in lastBonds[atom]:
                    for a2 in bondedTo[a1]:
                        newBonds[atom].add(a2)
            bondedWithSeparation.append(newBonds)

        # Build the list of pairs with the actual separation
        pairs = []
        for atom in range(numAtoms):
            for otherAtom in bondedWithSeparation[-1][atom]:
                if otherAtom > atom:
                    # Determine the minimum number of bonds between them
                    sep = maxSeparation
                    for i in reversed(range(maxSeparation-1)):
                        if otherAtom in bondedWithSeparation[i][atom]:
                            sep -= 1
                        else:
                            break
                    # Convert back to atom type strings
                    try:
                        drudeA = self.parent2drude[atoms[atom]]
                        drudeB = self.parent2drude[atoms[otherAtom]]
                    except KeyError:
                        # skip over atoms which don't have drudes (e.g., H atoms)
                        continue
                    thole = drudes[drudeA]['thole'] + drudes[drudeB]['thole']
                    screenedPair = (atom, otherAtom, thole)
                    pairs.append(screenedPair)

        return pairs
    
    def _drude_maps(self, drude_params):
        self.drude2parent = {}
        self.parent2drude = {}
        for key, dparam in drude_params.items():
            self.drude2parent[dparam['drude_type']] = dparam['parent_type']
            self.parent2drude[dparam['parent_type']] = dparam['drude_type']
    
    def parse_xml(self, xml_file):
        """Populate this XmlMD object by parsing an OpenMM-style XML file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Parse <AtomTypes>
        # <Type name="IM-N0"   class="NI0"  element="N" mass="13.6067"/>
        # <Type name="IM-DN0"  class="Sh"               mass="0"/>
        atomtypes_section = root.find('AtomTypes')
        if atomtypes_section is not None:
            for type_el in atomtypes_section.findall('Type'):
                name    = type_el.get('name')
                aclass  = type_el.get('class')
                elem    = type_el.get('element', '') # Drudes have no element
                mass = float(type_el.get('mass'))
                self.atom_types[name] = {
                    'class': aclass,
                    'element': elem,
                    'mass': mass
                }

        # Parse <Residues>
        # <Atom name="N00"  type="IM-N00"/>
        # NOTE: for OpenMM reproducibility, indicies are based on Residue topology!
        residues_section = root.find('Residues')
        if residues_section is not None:
            for residue_el in residues_section.findall('Residue'):
                rname = residue_el.get('name')
                atomlist = []
                for atom_el in residue_el.findall('Atom'):
                    aname = atom_el.get('name')
                    atype = atom_el.get('type')
                    atomlist.append((aname, atype))
                self.residues[rname] = atomlist
                
                # Parse Bond Topology
                # <Bond from="7"  to="6"/>
                for bond_el in residue_el.findall('Bond'):
                    b_from = int(bond_el.get('from'))
                    b_to   = int(bond_el.get('to'))

                    #self.bonds.append(Bond(b_from, b_to))
                    #self.bonds.append((bond_el.attrib['from'], bond_el.attrib['to']))
                    self.bonds.append((atomlist[b_from][1],atomlist[b_to][1]))

        # If more than one residue name is found, raise error
        if len(self.residues) > 1:
            raise ValueError(
                f"Multiple residues found ({list(self.residues.keys())}), "
                "but only homo-nmers are supported."
            )
        
        # Parse <NonbondedForce>
        # <Atom type="IM-N0"   charge="0.4939"  sigma="1.00000" epsilon="0.00000"/>
        nb_section = root.find('NonbondedForce')
        if nb_section is not None:
            for atom_el in nb_section.findall('Atom'):
                tname = atom_el.get('type')
                q     = float(atom_el.get('charge'))
                sig   = float(atom_el.get('sigma'))
                eps   = float(atom_el.get('epsilon'))
                self.nonbonded_params[tname] = (q, sig, eps)

        # Parse <DrudeForce>
        # <Particle type1="IM-DC21" type2="IM-C21" charge="-1.1478" polarizability="0.00195233" thole="1"/>
        drude_section = root.find('DrudeForce')
        if drude_section is not None:
            for part_el in drude_section.findall('Particle'):
                dtype = part_el.get('type1')
                ptype = part_el.get('type2')
                dq    = float(part_el.get('charge'))
                alpha = float(part_el.get('polarizability'))
                thole = float(part_el.get('thole'))
                self.drude_params[dtype] = {
                    'drude_type': dtype,
                    'parent_type': ptype,
                    'drude_charge': dq,
                    'polarizability': alpha,
                    'thole': thole
                }
        self.core_atom_types = [atype for aname, atype in self.residues[rname] if atype not in self.drude_params]
        self._drude_maps(drude_params=self.drude_params)
        self.screenedPairs = self._findExclusions(
            bonds=self.bonds,
            atoms=self.core_atom_types,
            drudes=self.drude_params,
        )

    def summary(self):
        """A demo method to show what's been parsed."""
        print(f"AtomTypes: {len(self.atom_types)} types parsed")
        print(f"All Atom Types: {self.atom_types}")
        print(f"Core Atom Types: {self.core_atom_types}")
        print(f"Residues:  {len(self.residues)} residue templates parsed")
        print(f"Bonds: {self.bonds}")
        print(f"Nonbonded: {len(self.nonbonded_params)} parameter entries")
        print(f"Drude:     {len(self.drude_params)} drude entries")
        print(f"Screened Pairs: {len(self.screenedPairs)}\n{self.screenedPairs}")
        if self.qcel_mol is not None:
            print("QCElemental Molecule is attached.")
            # e.g., print info about self.qcel_mol
            print(f"  Molecule name: {getattr(self.qcel_mol, 'name', '???')}")
            print(f"  Number of molecules: {len(self.qcel_mol.fragments)}")
            print(f"  Number of atoms: {len(self.qcel_mol.symbols)}")

        if self.atom_types_map is not None:
            print(f"Custom atom_types_map provided with {len(self.atom_types_map)} entries.")

def _map_mol(mol, _map):

    atom_types = pd.read_csv(_map, names=["From", "To"])
    source_type_symbols = [i[0] for i in atom_types["From"].values]
    n_fragments = len(mol.fragments)
    mapped_fragments = []

    for i in range(n_fragments):
        
        mol_i = mol.get_fragment(i)
        #print(mol_i.geometry)
        target_types = [f"{s}{i}" for i,s in enumerate(mol_i.symbols)]
        
        geometry = np.array(mol_i.geometry)*constants.conversion_factor("bohr", "nanometer")
        geometry_mapped = np.zeros_like(geometry)
        for idx, _type in enumerate(target_types):
            idx_mapped = atom_types[atom_types["To"] == _type].index[0]
            geometry_mapped[idx_mapped] = geometry[idx]
            
        mol_mapped_i = qcel.models.Molecule(
                symbols=source_type_symbols,
                geometry=geometry_mapped * constants.conversion_factor("nanometer", "bohr"),  # Convert back to bohr for QCElemental
                name=f"mapped_mol_{i}",
        )
        #print(mol_mapped_i)
        #print(mol_mapped_i.geometry)
        mapped_fragments.append(mol_mapped_i)
    
    combined_geoms = np.vstack([frag.geometry for frag in mapped_fragments])
    #print(combined_geoms)
    combined_symbols = []
    for frag in mapped_fragments:
        combined_symbols.extend(frag.symbols)

    # Create fragments list for the combined molecule
    fragments = []
    atom_counter = 0
    for frag in mapped_fragments:
        n_atoms = len(frag.symbols)
        fragments.append(list(range(atom_counter, atom_counter + n_atoms)))
        atom_counter += n_atoms
    
    # Create the combined molecule
    combined_mol = qcel.models.Molecule(
        symbols=combined_symbols,
        geometry=combined_geoms,
        name="mapped_mol_combined",
        fragments=fragments  # Properly specify fragments
    )

    return combined_mol


def _create_topology(qcel_mol, old_pdb_path, atom_types_map):
    """Create new topology based on QCElemental "topology"."""
    
    import warnings
    # suppress some MDAnalysis warnings about PDB files
    warnings.filterwarnings('ignore')

    resnames = set(Universe(old_pdb_path).atoms.resnames)
    # NOTE: All resiudes MUST be the same in the provided PDB topology 
    residue_name = resnames.pop() if len(resnames) == 1 else None
    
    old_pdb_path = Path(old_pdb_path)
    tmp_pdb = old_pdb_path.with_name(old_pdb_path.stem + "_tmp" + old_pdb_path.suffix)
   
    qcel_mol = _map_mol(qcel_mol, atom_types_map)
    _molecule_to_pdb_file(qcel_mol, tmp_pdb, residue_name, atom_types_map) 
    _add_CONECT(tmp_pdb)

    return str(tmp_pdb)

def _molecule_to_pdb_file(molecule, filename: str, res_name: str, atom_types_map: str | None) -> None:
    """Writes a QCElemental Molecule to a PDB file, handling multiple fragments and adding CONECT records if connectivity information is provided."""
    coords = molecule.geometry * constants.bohr2angstroms
    symbols = molecule.symbols
    fragments = molecule.fragments

    pdb_lines = []
    atom_idx = 1

    for res_seq, fragment in enumerate(fragments, start=1):
        residue_name = res_name
        chain_id = ' '
        if atom_types_map:
            type_map = pd.read_csv(atom_types_map, names=["From", "To"])#.set_index("To")
            type_map = type_map["From"].tolist()
        for i, atom in enumerate(fragment):
            symbol = symbols[atom]
            xyz = coords[atom]
            atom_name = type_map[i] #.loc[f"{symbol.upper()}{i}", "From"]

            pdb_lines.append(
                f"ATOM  {atom_idx:5d} {atom_name:<4} {residue_name:<3} {chain_id}{res_seq:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            )
            atom_idx += 1

    # pdb_lines.append("END")

    if molecule.connectivity:
        unique_bonds = set()
        for bond in molecule.connectivity:
            idx1, idx2 = sorted(bond[:2])  # QCElemental indices start from 0
            unique_bonds.add((idx1 + 1, idx2 + 1))

        for idx1, idx2 in unique_bonds:
            pdb_lines.append(f"CONECT{idx1:5d}{idx2:5d}")

    with open(filename, "w") as pdb:
        pdb.write('\n'.join(pdb_lines))

def _add_CONECT(pdb_filename: str) -> None:
    """Adds CONECT records to an existing PDB file using MDAnalysis's default bond guesser."""
    from MDAnalysis import Universe
    u = Universe(pdb_filename, format='PDB', guess_bonds=True)
    conect_lines = set()
    
    for bond in u.bonds:
        idx1, idx2 = sorted(bond.indices + 1)  # PDB atom numbering starts at 1
        conect_lines.add(f"CONECT{idx1:5d}{idx2:5d}")

    with open(pdb_filename, "a") as pdb:
        pdb.write('\n' + '\n'.join(sorted(conect_lines)) + '\n')

def get_Dij_omm(simmd):
    system = simmd.system
    topology = simmd.topology
    positions = simmd.context.getState(getPositions=True).getPositions()

    # Identify the DrudeForce
    drude_forces = [f for f in system.getForces() if isinstance(f, DrudeForce)]
    if len(drude_forces) == 0:
        raise ValueError("No DrudeForce found in the system.")
    drude = drude_forces[0]

    # Build a map from parent (core) index --> Drude (shell) index in the System
    num_drudes = drude.getNumParticles()

    # For convenience, track all Drude indices in a set to help with skipping them later
    drude_particle_indices = set()
    parent_to_drude = {}

    for i in range(num_drudes):
        # According to OpenMM docs, getParticleParameters(i) returns:
        # (drudeParticleIndex, parentIndex, charge, polarizability, aniso12, aniso13)
        #dIdx, pIdx, charge, pol, aniso12, aniso13 = drude.getParticleParameters(i)
        drudeParams = drude.getParticleParameters(i)
        dIdx = drudeParams[0]
        pIdx = drudeParams[1]
        drude_particle_indices.add(dIdx)
        parent_to_drude[pIdx] = dIdx

    # Now build per-residue arrays for r_core and r_shell.
    # We'll keep them parallel: the nth entry of each corresponds to the same atom in a residue.
    r_core_list = []
    r_shell_list = []

    for res in topology.residues():
        residue_core_pos = []
        residue_shell_pos = []
        for atom in res.atoms():
            # If this atom is itself a Drude particle, skip it for "core" arrays
            if atom.index in drude_particle_indices:
                continue

            # Core (parent) position
            core_pos = positions[atom.index]
            core_pos_nm = [p.value_in_unit(nanometer) for p in core_pos]

            # If there's a Drude shell for this parent, retrieve it; otherwise same as core or zero.
            if atom.index in parent_to_drude:
                shell_index = parent_to_drude[atom.index]
                shell_pos = positions[shell_index]
                shell_pos_nm = [p.value_in_unit(nanometer) for p in shell_pos]
            else:
                # No Drude => zero displacement.  You could just use core_pos if you want 
                # r_shell = r_core for non-Drude atoms, or put zero, etc.
                shell_pos_nm = core_pos_nm  # or [0.0, 0.0, 0.0] if you want to highlight no Drude

            residue_core_pos.append(core_pos_nm)
            residue_shell_pos.append(shell_pos_nm)

        r_core_list.append(residue_core_pos)
        r_shell_list.append(residue_shell_pos)

    # Convert to JAX arrays
    r_core = jnp.array(r_core_list)
    r_shell = jnp.array(r_shell_list)

    # Now compute the per-atom Drude displacement: shell - core
    return get_Dij(r_core, r_shell)

def get_Dij(r_core, r_shell):
    """Calculate displacement between core and shell particles."""
    shell_mask = safe_norm(r_shell, 0.0, axis=-1) > 0.0
    d = r_core - r_shell
    d = jnp.where(shell_mask[..., jnp.newaxis], d, 0.0)
    return d

def get_Rij_Dij(simmd=None,qcel_mol=None,atom_types_map=None, **kwargs):
    """Obtain Rij matrix (core-core displacements) and Dij (core-shell) displaments."""

    if simmd is not None:
        system = simmd.system
        topology = simmd.topology
        positions = simmd.context.getState(getPositions=True).getPositions()

        drude = [f for f in system.getForces() if isinstance(f, DrudeForce)][0]
        numDrudes = drude.getNumParticles()
        drude_indices = [drude.getParticleParameters(i)[0] for i in range(numDrudes)]

        r_core = []
        for i, res in enumerate(topology.residues()):
            residue_core_pos = []
            for atom in res.atoms():
                # skip over drude particles
                if atom.index in drude_indices:
                    continue
                pos = list(positions[atom.index])
                pos = [p.value_in_unit(nanometer) for p in pos]
                # update positions for residue
                residue_core_pos.append(pos)
            r_core.append(residue_core_pos)

        # conveniently, r_core = r_shell (i.e., initialize Dij to zero)
        r_core = jnp.array(r_core)
        r_shell = jnp.array(r_core)

        # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
        Rij = (
            r_core[jnp.newaxis, :, jnp.newaxis, :, :]
            - r_core[:, jnp.newaxis, :, jnp.newaxis, :]
        )
        Dij = get_Dij(r_core, r_shell)

        return Rij, Dij
    elif (simmd is None) and ((qcel_mol is not None) and (atom_types_map is not None)):
        nmols = len(qcel_mol.fragments)
        m = [] 
        for i in range(nmols):
            mi = qcel_mol.get_fragment(i)
            
            # atom_types_map ensure 1-to-1 mappings to SAPT "labels"
            atom_types = pd.read_csv(atom_types_map, names=["From", "To"])
            target_types = [f"{s}{i}" for i,s in enumerate(mi.symbols)]

            # NOTE: there should be a better way to determine qcel position units
            geometry = np.array(mi.geometry)*constants.conversion_factor("bohr", "nanometer")
            geometry_mapped = np.zeros_like(geometry)
            #print(atom_types["From"].values)
            #print(atom_types["To"].values)
            for idx, _type in enumerate(target_types):
                #print(_type)
                #print(atom_types[atom_types["To"] == _type].index)
                idx_mapped = atom_types[atom_types["To"] == _type].index[0]
                geometry_mapped[idx_mapped] = geometry[idx]
            
            m.append(geometry_mapped)
            #print(target_types)
            #print(mi.geometry*constants.conversion_factor("bohr", "nanometer"))
            #print(atom_types["From"].values)
            #print(geometry_mapped)
        r_core = jnp.stack(m)
        if kwargs.get("pdb_template") is not None:
            pdb_template = kwargs.get("pdb_template")
            r_core_to_pdb(r_core, pdb_template=pdb_template)

        r_shell = jnp.array(r_core)

        # broadcast r_core (nmols, natoms, 3) --> Rij (nmols, nmols, natoms, natoms, 3)
        Rij = (
            r_core[jnp.newaxis, :, jnp.newaxis, :, :]
            - r_core[:, jnp.newaxis, :, jnp.newaxis, :]
        )
        Dij = get_Dij(r_core, r_shell)
        return Rij, Dij

def r_core_to_pdb(r_core, pdb_template, pdb_file="tmp.pdb"):
    """Create updated PDB file with r_core from residue file as template."""

    pdb = PDBFile(pdb_template)

    # reshape r_core to match total number of atoms
    coords = np.array(r_core).reshape((-1, 3))

    # create list of Vec3 positions (default to nm)
    if coords.shape[0] != len(pdb.positions):
        raise ValueError(
            f"Mismatch in atom count: Template has {len(pdb.positions)} atoms "
            f"but r_core has {coords.shape[0]} coords."
        )
    new_positions = [Vec3(*xyz) for xyz in coords] * nanometer

    # write the updated PDB
    pdb.positions = new_positions
    with open(pdb_file, "w") as f:
        PDBFile.writeFile(pdb.topology, pdb.positions, f)
    
    print(f"Created {pdb_file}!") 

def get_QiQj(simmd):
    """Obtain core and shell charges.

    TODO: This information is centrally contained in the NonbondedForce class, i.e.

    <NonbondedForce coulomb14scale="0" lj14scale="0">
    <Atom type="acnt-CT" charge="1.263" sigma="1.00000" epsilon="0.00000"/>
    <Atom type="acnt-DCT" charge="-1.252" sigma="1.00000" epsilon="0.00000"/>
    ...
    </NonbondedForce>

    and Qi Qj terms can be created w/o OpenMM.

    """
    system = simmd.system
    topology = simmd.topology

    drude = [f for f in system.getForces() if isinstance(f, DrudeForce)][0]
    numDrudes = drude.getNumParticles()
    drude_indices = [drude.getParticleParameters(i)[0] for i in range(numDrudes)]
    parent_indices = [drude.getParticleParameters(i)[1] for i in range(numDrudes)]

    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    
    q_core = []
    q_shell = []
    for i, res in enumerate(topology.residues()):
        res_charge = []
        res_shell_charge = []
        #print(f"Setting up charges for {res.name}")
        for atom in res.atoms():
            #print(atom.index, atom.name)
            # skip over drude particles
            if atom.index in drude_indices:
                #print(f"*skipped {atom.name}*")
                continue
            charge, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
            #print(f"q={charge} for ({atom.index},{atom.name})")
            charge = charge.value_in_unit(elementary_charge)
            #print(f"q={charge} for ({atom.index},{atom.name})")
            # assign drude positions for respective parent atoms
            if atom.index in parent_indices:
                drude_params = drude.getParticleParameters(
                    parent_indices.index(atom.index)
                )
                drude_charge = drude_params[5].value_in_unit(elementary_charge)

                res_shell_charge.append(drude_charge)
            else:
                res_shell_charge.append(0.0)

            res_charge.append(charge)
        
        #print(res_charge)
        q_core.append(res_charge)
        q_shell.append(res_shell_charge)

    q_core = jnp.array(q_core)
    q_shell = jnp.array(q_shell)

    # break up core-shell, shell-core, and shell-shell terms
    Qi_shell = q_shell[:, jnp.newaxis, :, jnp.newaxis]
    Qj_shell = q_shell[jnp.newaxis, :, jnp.newaxis, :]
    Qi_core = q_core[:, jnp.newaxis, :, jnp.newaxis]
    Qj_core = q_core[jnp.newaxis, :, jnp.newaxis, :]

    return Qi_core, Qi_shell, Qj_core, Qj_shell


def get_pol_params(simmd):
    """Obtain spring constants and Thole screening term.

    Spring constants are defined as:
    k = q_shell^2 / alpha,
    where alpha are the atomic polarizabilities.

    The Thole screening term (later used to define the screening function, Sij) is:
    u_scale = a / (alpha_i * alpha_j)^(1/6),
    where "a" is the Thole damping constant.

    TODO: This information is centrally contained in the DrudeForce class, i.e.

    <DrudeForce>
     <Particle type1="acnt-DNZ" type2="acnt-NZ" charge="-1.015" polarizability="0.001527" thole="1"/>
    ...
    </DrudeForce>

    and k, u_scale terms can be created w/o OpenMM.
    """

    system = simmd.system
    topology = simmd.topology

    drude = [f for f in system.getForces() if isinstance(f, DrudeForce)][0]
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    numDrudes = drude.getNumParticles()

    drude_indices = [drude.getParticleParameters(i)[0] for i in range(numDrudes)]
    parent_indices = [drude.getParticleParameters(i)[1] for i in range(numDrudes)]

    q_shell = []
    alphas = []
    tholes = []
    tholeMatrixMade = False
    numResidues = len(list(topology.residues()))

    for i, res in enumerate(topology.residues()):
        res_shell_charge = []
        res_alpha = []
        numAtoms = len(list(res.atoms()))
        for atom in res.atoms():
            # assign drude positions for respective parent atoms
            if atom.index in drude_indices:
                continue
            charge, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
            charge = charge.value_in_unit(elementary_charge)
            if atom.index in parent_indices:
                # map parent index to drude index
                drude_params = drude.getParticleParameters(
                    parent_indices.index(atom.index)
                )
                drude_charge = drude_params[5].value_in_unit(elementary_charge)
                alpha = drude_params[6]
                numScreenedPairs = drude.getNumScreenedPairs()
                if numScreenedPairs > 0:
                    if not tholeMatrixMade:
                        natoms_per_res = int(
                            (topology.getNumAtoms() - len(drude_indices))
                            / topology.getNumResidues()
                        )
                        natoms = len(list(res.atoms()))
                        nmol = len(list(topology.residues()))
                        tholeMatrix = np.zeros(
                            (nmol, natoms_per_res, natoms_per_res)
                        )  # this assumes that the u_scale term is identical between core-core, shell-shell, and core-shell interactions

                        for sp_i in range(numScreenedPairs):
                            screened_params = drude.getScreenedPairParameters(sp_i)
                            #print(f"sp_{sp_i}: {screened_params}")
                            prt0_params = drude.getParticleParameters(
                                screened_params[0]
                            )
                            core0 = prt0_params[1]
                            alpha0 = prt0_params[6].value_in_unit(nanometer**3)
                            imol = int(core0 / natoms)
                            prt1_params = drude.getParticleParameters(
                                screened_params[1]
                            )
                            core1 = prt1_params[1]
                            alpha1 = prt1_params[6].value_in_unit(nanometer**3)
                            thole = screened_params[2]
                            # ensure indices don't exceed single-residue atom indices
                            if core0 >= natoms:
                                core0 = core0 % natoms
                            if core1 >= natoms:
                                core1 = core1 % natoms

                            tholeMatrix[imol][core0][core1] = thole / (
                                alpha0 * alpha1
                            ) ** (1.0 / 6.0)
                            tholeMatrix[imol][core1][core0] = thole / (
                                alpha0 * alpha1
                            ) ** (1.0 / 6.0)

                        tholeMatrix = list(tholeMatrix)
                        tholeMatrixMade = True
                elif numScreenedPairs == 0:
                    tholeMatrixMade = False

                res_shell_charge.append(drude_charge)
            else:
                res_shell_charge.append(0.0)
                alpha = 0.0 * nanometer**3
            alpha = alpha.value_in_unit(nanometer**3)

            # update positions for residue
            res_alpha.append(alpha)

        q_shell.append(res_shell_charge)
        alphas.append(res_alpha)

    q_shell = jnp.array(q_shell)
    alphas = jnp.array(alphas)

    _alphas = jnp.where(alphas == 0.0, jnp.inf, alphas)
    k = jnp.where(alphas == 0.0, 0.0, ONE_4PI_EPS0 * q_shell**2 / _alphas)
    if tholeMatrixMade:
        tholes = jnp.array(tholeMatrix)
        u_scale = (
            tholes[jnp.newaxis, ...]
            * jnp.eye(numResidues)[:, :, jnp.newaxis, jnp.newaxis]
        )
    else:
        tholes = jnp.zeros((numResidues, numResidues, numAtoms))
        u_scale = 0.0  # tholes * jnp.eye(Rij.shape[0])[:,:,jnp.newaxis,jnp.newaxis]

    return k, u_scale


def setup_openmm(
    pdb_file,
    ff_file,
    residue_file,
    timestep=0.00001 * picoseconds,
    error_tol=0.0001,
    integrator_seed=None,
    platform_name="CPU",
):
    """
    Function to create Simulation object from OpenMM.

    Arguments:
    <str> pdb_file
        .pdb for creating simulation topology & obtaining positions
    <str> ff_file
        .xml with force field parameters
    <str> residue_file
        .xml for non-standard residue topology
    """

    # obtain bond definitions and atom/Drude positions
    Topology().loadBondDefinitions(residue_file)
    integrator = DrudeSCFIntegrator(timestep)
    integrator.setMinimizationErrorTolerance(error_tol)
    if integrator_seed is not None:
        integrator.setRandomNumberSeed(integrator_seed)
    pdb = PDBFile(pdb_file)
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField(ff_file)

    modeller.addExtraParticles(forcefield)

    system = forcefield.createSystem(
        modeller.topology, constraints=None, rigidWater=True
    )

    for i in range(system.getNumForces()):
        f = system.getForce(i)
        f.setForceGroup(i)

    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    # Add exceptions for ALL intramolecular pairs in a residue
    for residue in modeller.getTopology().residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        for i in range(len(atom_indices)):
            for j in range(i + 1, len(atom_indices)):
                i_global = atom_indices[i]
                j_global = atom_indices[j]
                # Force the Coulomb & LJ to zero for i-j
                nonbonded.addException(i_global, j_global, 0.0, 1.0, 0.0, True)

    platform = Platform.getPlatformByName(platform_name)
    simmd = Simulation(modeller.topology, system, integrator, platform)
    simmd.context.setPositions(modeller.positions)

    return simmd


def U_ind_omm(simmd, decomp=False):
    # total *static* energy (i.e., while Drudes have zero contribution)
    state = simmd.context.getState(
        getEnergy=True, getForces=True, getVelocities=True, getPositions=True
    )
    U_static_omm = state.getPotentialEnergy()

    # optimize Drude positions
    simmd.step(1)
    state = simmd.context.getState(
        getEnergy=True, getForces=True, getVelocities=True, getPositions=True
    )

    # total Nonbonded + Drude (self) energy
    U_tot_omm = state.getPotentialEnergy()
    Uind_omm = (U_tot_omm - U_static_omm).value_in_unit(kilojoules_per_mole)
    if decomp:
        system = simmd.system
        for j in range(system.getNumForces()):
            f = system.getForce(j)
            PE = str(type(f)) + str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy())
            if 'NonbondedForce' in PE:
                _NonbondedForce = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                _NonbondedForce = _NonbondedForce.value_in_unit(kilojoules_per_mole)
            if 'DrudeForce' in PE:
                _DrudeForce = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                _DrudeForce = _DrudeForce.value_in_unit(kilojoules_per_mole)
        return Uind_omm, _DrudeForce, _NonbondedForce
    else:
        return Uind_omm

def get_QiQj_off(xmlmd):
    """Obtain core and shell charges."""

    q_core = []
    q_shell = []
    nmols = len(xmlmd.qcel_mol.fragments)
    for i in range(nmols):
        res_charge = [] 
        res_shell_charge = []
        resname = next(iter(xmlmd.residues))
        for _, atom in xmlmd.residues[resname]:
            if xmlmd.atom_types[atom]["class"] == "Sh":
                # skip over drude particles
                continue 
            # assign drude charges for respective parent atoms
            if atom in xmlmd.parent2drude:
                drude_atom = xmlmd.parent2drude[atom]
                res_shell_charge.append(xmlmd.drude_params[drude_atom]['drude_charge'])
            else: 
                res_shell_charge.append(0.0)
            q, _, _ = xmlmd.nonbonded_params[atom]
            res_charge.append(q)
        q_core.append(res_charge)
        q_shell.append(res_shell_charge)
    
    q_core = jnp.array(q_core)
    q_shell = jnp.array(q_shell)
    
    # break up core-shell, shell-core, and shell-shell terms
    Qi_shell = q_shell[:, jnp.newaxis, :, jnp.newaxis]
    Qj_shell = q_shell[jnp.newaxis, :, jnp.newaxis, :]
    Qi_core = q_core[:, jnp.newaxis, :, jnp.newaxis]
    Qj_core = q_core[jnp.newaxis, :, jnp.newaxis, :]

    return Qi_core, Qi_shell, Qj_core, Qj_shell

def get_pol_params_off(xmlmd):
    """Obtain spring constants and Thole screening term.

    Spring constants are defined as:
    k = q_shell^2 / alpha,
    where alpha are the atomic polarizabilities.

    The Thole screening term (later used to define the screening function, Sij) is:
    u_scale = a / (alpha_i * alpha_j)^(1/6),
    where "a" is the Thole damping constant.
    """
    
    # initialize polarizable parameters 
    alphas = []
    q_shell = []
    nmols = len(xmlmd.qcel_mol.fragments)
    # assume `u_scale` is identical between core-core, shell-shell, and core-shell terms
    tholeMatrix = np.zeros(
            (nmols, len(xmlmd.core_atom_types), len(xmlmd.core_atom_types))
            )

    # create *ONLY* intra-molecular screening terms for every molecule
    for i in range(nmols):
        tholeMatrixMade = False # only create tholeMatrix for each molecule once
        mol_q_shell = []
        mol_alpha = []
        resname = next(iter(xmlmd.residues)) # NOTE: there should only be one residue
        for _, atom in xmlmd.residues[resname]: 
            if xmlmd.atom_types[atom]["class"] == "Sh":
                # skip over drude particles 
                continue 
            if atom in xmlmd.parent2drude: # or in xmlmd.core_atom_types
                drude_charge = xmlmd.drude_params[xmlmd.parent2drude[atom]]["drude_charge"]
                alpha = xmlmd.drude_params[xmlmd.parent2drude[atom]]["polarizability"]
                if len(xmlmd.screenedPairs) > 0 and not tholeMatrixMade: 
                    for j, sp in enumerate(xmlmd.screenedPairs):
                        drudei = xmlmd.parent2drude[xmlmd.core_atom_types[sp[0]]]
                        drudej = xmlmd.parent2drude[xmlmd.core_atom_types[sp[1]]]
                        
                        alphai = xmlmd.drude_params[drudei]["polarizability"]
                        alphaj = xmlmd.drude_params[drudej]["polarizability"]
                        
                        thole = sp[2]

                        tholeMatrix[i][sp[0]][sp[1]] = thole / (
                                alphai * alphaj
                        ) ** (1.0 / 6.0)
                        tholeMatrix[i][sp[1]][sp[0]] = thole / (
                                alphai * alphaj
                        ) ** (1.0 / 6.0)
                    # only need to explore thole once per molecule 
                    tholeMatrixMade = True
                else:
                    if len(xmlmd.screenedPairs) == 0:
                        print(f"No screenedPairs found!!!")
                mol_q_shell.append(drude_charge)
            else:
                mol_q_shell.append(0.0)
                alpha = 0.0
            mol_alpha.append(alpha)
        q_shell.append(mol_q_shell)
        alphas.append(mol_alpha)
   
    q_shell = jnp.array(q_shell)
    alphas = jnp.array(alphas)
    _alphas = jnp.where(alphas == 0.0, jnp.inf, alphas)
    k = jnp.where(alphas == 0.0, 0.0, ONE_4PI_EPS0 * q_shell**2 / _alphas)

    if tholeMatrixMade:
        tholes = jnp.array(tholeMatrix)
        u_scale = (
            tholes[jnp.newaxis, ...]
            * jnp.eye(nmols)[:, :, jnp.newaxis, jnp.newaxis]
        )
    else: # e.g., H2O 
        tholes = jnp.zeros((nmols, nmols, len(xmlmd.core_atom_types)))
        u_scale = 0.0  # tholes * jnp.eye(Rij.shape[0])[:,:,jnp.newaxis,jnp.newaxis]

    return k, u_scale
