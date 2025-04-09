import os
import jax.numpy as jnp
from optax import safe_norm
import time
import qcelemental as qcel
from qcelemental import constants
import numpy as np
import pandas as pd
from pathlib import Path
from MDAnalysis import Universe 

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
            type_map = pd.read_csv(atom_types_map, names=["From", "To"]).set_index("To")
             
        for i, atom in enumerate(fragment):
            symbol = symbols[atom]
            xyz = coords[atom]
            atom_name = type_map.loc[f"{symbol.upper()}{i}", "From"]

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
    Universe(pdb_filename, format='PDB', guess_bonds=True).atoms.write(pdb_filename)


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
        for atom in res.atoms():
            #print(atom.index, atom.name)
            # skip over drude particles
            if atom.index in drude_indices:
                continue
            charge, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
            charge = charge.value_in_unit(elementary_charge)
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


def U_ind_omm(simmd):
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
    return (U_tot_omm - U_static_omm).value_in_unit(kilojoules_per_mole)
