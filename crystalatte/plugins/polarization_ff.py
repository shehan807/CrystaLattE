from . import utils
from optax import safe_norm
from jaxopt import NonlinearCG
import jax.numpy as jnp
import jax
from jax import jit
import qcelemental as qcel
import itertools
from collections import Counter
from pathlib import Path

import pandas as pd

# NOTE: process-wide; the Drude SCF needs float64 precision.
jax.config.update("jax_enable_x64", True)

ONE_4PI_EPS0 = utils.ONE_4PI_EPS0


@jit
def jnp_denominator_norm(X):
    """Enable nan-friendly gradients & divide by zero."""
    X_norm = safe_norm(X, 0.0, axis=-1)
    return jnp.where(X_norm == 0.0, jnp.inf, X_norm)

@jit
def safe_sum(X):
    """Enable safe sum for jnp matrices with infty."""
    return jnp.where(jnp.isfinite(X), X, 0).sum()

@jit
def _DrudeForce(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k):
    """Direct `DrudeForce` comparison with OpenMM DrudeForce."""
    (nmol, _, natoms, _, pos) = Rij.shape
    if Dij.shape != (nmol, natoms, pos):
        Dij = jnp.reshape(Dij, (nmol, natoms, pos))
    U_coul_intra = Ucoul_intra(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    U_self       = Uself(Dij, k)
    return U_coul_intra + U_self

@jit
def _NonbondedForce(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    """Direct `NonbondedForce` comparison with OpenMM DrudeForce."""
    (nmol, _, natoms, _, pos) = Rij.shape
    if Dij.shape != (nmol, natoms, pos):
        Dij = jnp.reshape(Dij, (nmol, natoms, pos))
    U_coul       = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    U_coul_intra = Ucoul_intra(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    return U_coul - U_coul_intra 

@jit
def make_Sij(Rij, u_scale):
    """Build Thole screening function for intra-molecular dipole-dipole interactions.

    The Thole screening function S_ij is given by:
    
    S_ij = 1 - (1 + (u_scale * R_ij)/2) * exp(-u_scale * R_ij),
    
    where:
    - R_ij is the distance matrix between sites i and j
    - u_scale is the screening parameter (units: 1/distance) (See utils.get_pol_params)
    
    u_scale = a / (alpha_i * alpha_j)^(1/6),

    where:
    - a is the damping constant, i.e. the Thole parameter
    - alpha_i and alpha_j are the atomic polarizabilities for atoms i and j

    The atomic polarizability is derived from

    alpha = q_Drude^2 / k, 
    
    where: 
    - q_Drude is the Drude oscillator charge 
    - k is the force constant of the Drude-atom harmonic bond
    """
    Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    return 1.0 - (1.0 + 0.5 * Rij_norm * u_scale) * jnp.exp(-u_scale * Rij_norm)

@jit
def Uself(Dij, k):
    """Calculates self energy due to harmonic potentials of Drude springs.
    
    U_self = (1/2) * Σ_i k_i * ||d_i||^2,

    where 
    - k_i is the force constant
    - d_i is the displacement of the mobile Drude particle from the position of 
      the parent atom's nucleus
    """
    d_mag = safe_norm(Dij, 0.0, axis=2)
    return 0.5 * jnp.sum(k * d_mag**2)

@jit
def Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core):
    """Compute static Coulomb energy (i.e., q = q_shell + q_core).

    U_coul_static = (1/4πε₀) * (1/2) * ∑∑_{i≠j} Qi * Qj / |R_ij|,

    where:
    - 1/4πε₀ is the Coulomb constant
    - Qi, Qj are the static charges, 
      Qi=(Qi_core + Qi_shell) and Qj=(Qi_core + Qi_shell)
    - R_ij is the distance matrix between sites i and j
    """ 
    
    Rij_norm = jnp_denominator_norm(Rij)           
    U_coul_static = (Qi_core + Qi_shell) * (Qj_core + Qj_shell) / Rij_norm

    # remove intramolecular contributions
    I = jnp.eye(U_coul_static.shape[0])
    U_coul_static = U_coul_static * (1 - I[:, :, jnp.newaxis, jnp.newaxis])
    U_coul_static = 0.5 * safe_sum(U_coul_static)

    return ONE_4PI_EPS0 * U_coul_static

@jit
def Ucoul_intra(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    """Compute (damped) Drude-Drude, intra-molecular Coulomb energy.

    Damped, induced dipole-induced dipole interactions are derived from screening
    Coulomb interactions between specific pairs of dipoles:

    U_coul_intra = (1/4πε₀) * (1/2) * ∑∑_{i≠j} [
                    Qi_shell * Qj_shell / |R_ij|
                  + Qi_shell * Qj_shell / |R_ij + D_i|
                  + Qi_shell * Qj_shell / |R_ij + D_j|
                  + Qi_shell * Qj_shell / |R_ij + D_i - D_j|
                  ]

    where:
    - 1/4πε₀ is the Coulomb constant
    - Qi_shell, Qj_shell are the Drude charges on shared molecules 
    - R_ij is the distance matrix between sites i and j
    - D_i, D_j are the displacements of Drude particles from parent atom nuclei
    """ 
    
    # build denominator rij terms i
    Di = Dij[:, jnp.newaxis, :, jnp.newaxis, :]
    Dj = Dij[jnp.newaxis, :, jnp.newaxis, :, :]
    Rij_norm       = jnp_denominator_norm(Rij)           
    Rij_Di_norm    = jnp_denominator_norm(Rij + Di)      
    Rij_Dj_norm    = jnp_denominator_norm(Rij - Dj)      
    Rij_Di_Dj_norm = jnp_denominator_norm(Rij + Di - Dj) 

    # build Thole screening matrices
    Sij       = make_Sij(Rij, u_scale)       
    Sij_Di    = make_Sij(Rij + Di, u_scale)    
    Sij_Dj    = make_Sij(Rij - Dj, u_scale)    
    Sij_Di_Dj = make_Sij(Rij + Di - Dj, u_scale) 

    # compute intramolecular Coulomb matrix (of screened dipole-dipole pairs)
    U_coul_intra = (
            Sij       *  Qi_shell *  Qj_shell / Rij_norm
          - Sij_Di    *  Qi_shell *  Qj_shell / Rij_Di_norm
          - Sij_Dj    *  Qi_shell *  Qj_shell / Rij_Dj_norm
          + Sij_Di_Dj *  Qi_shell *  Qj_shell / Rij_Di_Dj_norm
    )
    # keep diagonal (intramolecular) components except for self-terms
    I_intra = jnp.eye(U_coul_intra.shape[0])
    I_self = jnp.eye(U_coul_intra.shape[-1])
    U_coul_intra = (U_coul_intra * I_intra[:, :, jnp.newaxis, jnp.newaxis]) * (
        1 - I_self[jnp.newaxis, jnp.newaxis, :, :]
    )
    U_coul_intra = 0.5 * safe_sum(U_coul_intra)
    
    return ONE_4PI_EPS0 * U_coul_intra

@jit
def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    """Compute total Coulomb energy.

    The total Coulomb energy is comprised of (1) U_coul_inter, i.e., all 
    inter-molecular site-site (Drude or parent) interactions and (2) U_coul_intra, 
    i.e., all intra-molecular, (screened) Drude-Drude interactions, 
    
    U_coul = U_coul_inter + U_coul_intra, 

    where

    U_coul_inter = (1/4πε₀) * (1/2) * ∑∑_{i≠j} [
                    Qi_core  * Qj_core  / |R_ij|
                  + Qi_shell * Qj_core  / |R_ij + D_i|
                  + Qi_core  * Qj_shell / |R_ij + D_j|
                  + Qi_shell * Qj_shell / |R_ij + D_i - D_j|
                  ]
    and 
    
    U_coul_intra = (1/4πε₀) * (1/2) * ∑∑_{i≠j} [
                    Qi_shell * Qj_shell / |R_ij|
                  - Qi_shell * Qj_shell / |R_ij + D_i|
                  - Qi_shell * Qj_shell / |R_ij + D_j|
                  + Qi_shell * Qj_shell / |R_ij + D_i - D_j|
                  ]

    where:
    - 1/4πε₀ is the Coulomb constant
    - Qi_shell, Qj_shell are the Drude charges (only for shared molecules for U_coul_intra)
    - Qi_core, Qj_core are the respective parent charges 
    - R_ij is the distance matrix between sites i and j
    - D_i, D_j are the displacements of Drude particles from parent atom nuclei
    """ 
    
    # build denominator rij terms i
    Di = Dij[:, jnp.newaxis, :, jnp.newaxis, :]
    Dj = Dij[jnp.newaxis, :, jnp.newaxis, :, :]
    Rij_norm       = jnp_denominator_norm(Rij)           
    Rij_Di_norm    = jnp_denominator_norm(Rij + Di)      
    Rij_Dj_norm    = jnp_denominator_norm(Rij - Dj)      
    Rij_Di_Dj_norm = jnp_denominator_norm(Rij + Di - Dj) 

    # build Thole screening matrices
    Sij       = make_Sij(Rij, u_scale)       
    Sij_Di    = make_Sij(Rij + Di, u_scale)    
    Sij_Dj    = make_Sij(Rij - Dj, u_scale)    
    Sij_Di_Dj = make_Sij(Rij + Di - Dj, u_scale) 

    # compute intermolecular Coulomb matrix
    U_coul_inter = (
            Qi_core  * Qj_core  / Rij_norm
          + Qi_shell * Qj_core  / Rij_Di_norm
          + Qi_core  * Qj_shell / Rij_Dj_norm
          + Qi_shell * Qj_shell / Rij_Di_Dj_norm
    )
    # remove diagonal (intramolecular) components
    # NOTE: ignores ALL nonbonded interactions for bonded atoms (i.e., 1-5, 1-6, etc.)
    I = jnp.eye(U_coul_inter.shape[0])
    U_coul_inter = U_coul_inter * (1 - I[:, :, jnp.newaxis, jnp.newaxis])

    # compute intramolecular Coulomb matrix (of screened dipole-dipole pairs)
    U_coul_intra = (
            Sij       * Qi_shell * Qj_shell / Rij_norm
          - Sij_Di    * Qi_shell * Qj_shell / Rij_Di_norm
          - Sij_Dj    * Qi_shell * Qj_shell / Rij_Dj_norm
          + Sij_Di_Dj * Qi_shell * Qj_shell / Rij_Di_Dj_norm
    )
    # keep diagonal (intramolecular) components except for self-terms
    I_intra = jnp.eye(U_coul_intra.shape[0])
    I_self = jnp.eye(U_coul_intra.shape[-1])
    U_coul_intra = (U_coul_intra * I_intra[:, :, jnp.newaxis, jnp.newaxis]) * (
        1 - I_self[jnp.newaxis, jnp.newaxis, :, :]
    )
    U_coul_total = 0.5 * safe_sum(U_coul_inter + U_coul_intra)
    
    return ONE_4PI_EPS0 * U_coul_total

@jit
def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k):
    """Calculate total induction energy.

    U_ind = (U_coul - U_coul_static) + U_self.

    Arguments:
    <jaxlib.xla_extension.ArrayImp> Rij (nmol, nmol, natoms, natoms, 3)
       JAX array of core-core atom x,y,z displacements 
    <jaxlib.xla_extension.ArrayImp> Dij (nmol, natoms, 3) or (nmol*natoms*3, )
       JAX array of core-shell atom x,y,z displacements )
    <jaxlib.xla_extension.ArrayImp> Qi_shell (nmol, 1, natoms, 1)
       JAX array of shell charges, row-wise
    <jaxlib.xla_extension.ArrayImp> Qj_shell (1, nmol, 1, natoms)
       JAX array of shell charges, column-wise
    <jaxlib.xla_extension.ArrayImp> Qi_core (nmol, 1, natoms, 1)
       JAX array of core charges, row-wise
    <jaxlib.xla_extension.ArrayImp> Qj_core (1, nmol, 1, natoms)
       JAX array of core charges, column-wise
    <jaxlib.xla_extension.ArrayImp> u_scale (nmol, nmol, natoms, natoms)
       JAX array of Thole screening term, a/(alphai*alphaj)^(1/6)
    <jaxlib.xla_extension.ArrayImp> k (nmol, natoms)
       JAX array of Drude spring constants, k = q_D^2 / alpha

    Returns:
    <np.float> Uind
        induction energy
    """
    (nmol, _, natoms, _, pos) = Rij.shape
    if Dij.shape != (nmol, natoms, pos):
        Dij = jnp.reshape(Dij, (nmol, natoms, pos))

    U_coul = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    U_coul_static = Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core)
    U_self = Uself(Dij, k)
    
    return (U_coul - U_coul_static) + U_self

@jit
def drudeOpt(
    Rij,
    Dij0,
    Qi_shell,
    Qj_shell,
    Qi_core,
    Qj_core,
    u_scale,
    k,
):
    """
    Iteratively determine core/shell displacements, d, by minimizing
    Uind w.r.t d.

    """

    Uind_min = lambda Dij: Uind(
        Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k
    )

    solver = NonlinearCG(fun=Uind_min, tol=1e-8, maxiter=500)
    res = solver.run(init_params=Dij0)
    return res.params

def cluster_induction_energy(qcel_mol, **kwargs):
    """
    Drude-oscillator induction energy of a single n-mer cluster.

    Parses the cluster topology and force-field parameters, relaxes the
    Drude (shell) displacements by SCF, and returns U_ind in kJ/mol. With
    omm_decomp=True, also returns the OpenMM-style Drude / nonbonded /
    static-Coulomb decomposition.
    """

    xml_file = kwargs.get("xml_file", None)
    atom_types_map = kwargs.get("atom_types_map", None)

    # fix topological issues in QCElemental (if any)
    qcel_mol = utils._fix_topological_order(qcel_mol)

    xmlmd = utils.XmlMD(qcel_mol=qcel_mol, atom_types_map=atom_types_map)
    xmlmd.parse_xml(xml_file)

    Rij, Dij = utils.get_Rij_Dij(qcel_mol=qcel_mol, atom_types_map=atom_types_map)
    Qi_core, Qi_shell, Qj_core, Qj_shell = utils.get_QiQj(xmlmd)
    k, u_scale = utils.get_pol_params(xmlmd)

    Dij = drudeOpt(
        Rij,
        jnp.ravel(Dij),
        Qi_shell,
        Qj_shell,
        Qi_core,
        Qj_core,
        u_scale,
        k,
    )
    
    U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)
    if kwargs.get("omm_decomp"):
        Ues = Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core)
        U_df = _DrudeForce(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)
        U_nb = _NonbondedForce(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
        return U_ind, U_df, U_nb, Ues
    else:
        return U_ind

_REQUIRED_KWARGS = {
    "xml_file": ".xml",
    "atom_types_map": ".csv",
}

def _check_inputs(qcel_mol, kwargs):
    """Validate user-supplied FF inputs, with guidance on how to fix them."""
    for key, suffix in _REQUIRED_KWARGS.items():
        path = kwargs.get(key)
        if path is None:
            raise ValueError(
                f"polarization_energy_function requires `{key}=<path to {suffix}>` "
                "passed as a keyword argument to crystalatte.main()."
            )
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"`{key}` not found: '{path}' (resolved from cwd '{Path.cwd()}'). "
                "Relative paths are resolved from the directory crystalatte.main() is run in."
            )
        if path.suffix.lower() != suffix:
            raise ValueError(f"`{key}` should be a {suffix} file, got '{path.name}'.")

    # Each monomer must match the atom-type map ("To" column = <element><index>).
    type_map = pd.read_csv(kwargs["atom_types_map"], names=["From", "To"])
    map_elems = Counter("".join(c for c in t if c.isalpha()).upper() for t in type_map["To"])
    for i, frag in enumerate(qcel_mol.fragments):
        frag_elems = Counter(qcel_mol.symbols[a].upper() for a in frag)
        if frag_elems != map_elems:
            raise ValueError(
                f"Monomer {i} composition {dict(frag_elems)} does not match `atom_types_map` "
                f"{dict(map_elems)}. Partial monomers usually mean the supercell is too small for "
                "`r_cut_monomer` (increase cif_a/cif_b/cif_c); otherwise check `bfs_thresh` and that "
                "the map lists every atom of the (single) molecule type."
            )

def polarization_energy_function(
    qcel_mol: qcel.models.Molecule,
    cif_output: str,
    nmers: dict,
    keynmer: str,
    nmer: dict,
    rminseps: str,
    rcomseps: str,
    cle_run_type: list,
    method="drude_oscillator",
    bsse_type=None,
    job_memory=None,
    verbose=0,
    **kwargs,
):
    """
    Drude-oscillator induction plugin for crystalatte.main(custom_function=...).

    Every crystalatte energy function plugin must accept the above arguments;
    `method`, `bsse_type`, and `job_memory` are accepted but ignored here.

    Required kwargs (forwarded from crystalatte.main()):
    - xml_file: OpenMM-style Drude FF parameters
    - atom_types_map: CSV mapping FF atom names to `<element><index>` of each monomer

    Stores the non-additive many-body induction energy, in Hartree, in
    `nmer["nambe"]` and returns it. Note that crystalatte's `output_data`
    reports `nambe` in kJ/mol.
    """
    _check_inputs(qcel_mol, kwargs)
    pol_kwargs = {
        "xml_file": kwargs.get("xml_file"),
        "atom_types_map": kwargs.get("atom_types_map"),
    }

    # Non-additive many-body induction by inclusion-exclusion over sub-clusters:
    #   ΔE(N) = Σ_{|S|≥2} (−1)^(N−|S|) U_ind(S);  1-body induction is zero, so omitted.
    n = len(nmer["monomers"])
    nambe = 0.0
    for k in range(2, n + 1):
        for subset in itertools.combinations(range(n), k):
            nambe += (-1) ** (n - k) * cluster_induction_energy(
                qcel_mol.get_fragment(list(subset)), **pol_kwargs
            )

    nmer["nambe"] = float(nambe) / qcel.constants.hartree2kJmol
    return nmer["nambe"]
