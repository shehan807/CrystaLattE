from . import openmm_utils
from optax import safe_norm
from jaxopt import BFGS
import jax.numpy as jnp
import numpy as np
import qcelemental as qcel
import os

# U_Pol START
import time
import sys

sys.path.append(".")


ONE_4PI_EPS0 = openmm_utils.ONE_4PI_EPS0

# @jit
def Uself(Dij, k):
    """
    Calculates self energy,
    U_self = 1/2 Σ k_i * ||d_mag_i||^2.

    Arguments:
    <np.array> Dij
        array of displacements between core and shell sites
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Uself
        polarization energy
    """
    d_mag = safe_norm(Dij, 0.0, axis=2)
    return 0.5 * jnp.sum(k * d_mag**2)


# @jit
def Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core):
    # use where solution to enable nan-friendly gradients
    Rij_norm = safe_norm(Rij, 0.0, axis=-1)

    # allow divide by zero
    _Rij_norm = jnp.where(Rij_norm == 0.0, jnp.inf, Rij_norm)

    U_coul_static = (Qi_core + Qi_shell) * (Qj_core + Qj_shell) / _Rij_norm

    # remove intramolecular contributions
    I = jnp.eye(U_coul_static.shape[0])
    mask = 1 - I[:, :, jnp.newaxis, jnp.newaxis]
    U_coul_static = U_coul_static * mask
    U_coul_static = (
        0.5 * jnp.where(jnp.isfinite(U_coul_static), U_coul_static, 0).sum()
    )  # might work in jax

    return ONE_4PI_EPS0 * (U_coul_static)


# @jit
def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    Di = Dij[:, jnp.newaxis, :, jnp.newaxis, :]
    Dj = Dij[jnp.newaxis, :, jnp.newaxis, :, :]

    # use where solution to enable nan-friendly gradients
    Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    Rij_Di_norm = safe_norm(Rij + Di, 0.0, axis=-1)
    Rij_Dj_norm = safe_norm(Rij - Dj, 0.0, axis=-1)
    Rij_Di_Dj_norm = safe_norm(Rij + Di - Dj, 0.0, axis=-1)

    # allow divide by zero
    _Rij_norm = jnp.where(Rij_norm == 0.0, jnp.inf, Rij_norm)
    _Rij_Di_norm = jnp.where(Rij_Di_norm == 0.0, jnp.inf, Rij_Di_norm)
    _Rij_Dj_norm = jnp.where(Rij_Dj_norm == 0.0, jnp.inf, Rij_Dj_norm)
    _Rij_Di_Dj_norm = jnp.where(Rij_Di_Dj_norm == 0.0, jnp.inf, Rij_Di_Dj_norm)

    # Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    Sij = 1.0 - (1.0 + 0.5 * Rij_norm * u_scale) * jnp.exp(-u_scale * Rij_norm)
    Sij_Di = 1.0 - (1.0 + 0.5 * Rij_Di_norm * u_scale) * \
        jnp.exp(-u_scale * Rij_Di_norm)
    Sij_Dj = 1.0 - (1.0 + 0.5 * Rij_Dj_norm * u_scale) * \
        jnp.exp(-u_scale * Rij_Dj_norm)
    Sij_Di_Dj = 1.0 - (1.0 + 0.5 * Rij_Di_Dj_norm * u_scale) * jnp.exp(
        -u_scale * Rij_Di_Dj_norm
    )

    # total coulomb energy
    U_coul = (
        Qi_core * Qj_core / _Rij_norm
        + Qi_shell * Qj_core / _Rij_Di_norm
        + Qi_core * Qj_shell / _Rij_Dj_norm
        + Qi_shell * Qj_shell / _Rij_Di_Dj_norm
    )

    # trying with safe_norm
    U_coul_intra = (
        Sij * -Qi_shell * -Qj_shell / _Rij_norm
        + Sij_Di * Qi_shell * -Qj_shell / _Rij_Di_norm
        + Sij_Dj * -Qi_shell * Qj_shell / _Rij_Dj_norm
        + Sij_Di_Dj * Qi_shell * Qj_shell / _Rij_Di_Dj_norm
    )

    # keep diagonal (intramolecular) components except for self-terms
    I_intra = jnp.eye(U_coul_intra.shape[0])
    I_self = jnp.eye(U_coul_intra.shape[-1])
    U_coul_intra = (U_coul_intra * I_intra[:, :, jnp.newaxis, jnp.newaxis]) * (
        1 - I_self[jnp.newaxis, jnp.newaxis, :, :]
    )
    U_coul_intra = (
        0.5 * jnp.where(jnp.isfinite(U_coul_intra), U_coul_intra, 0).sum()
    )  # might work in jax

    # remove diagonal (intramolecular) components
    # note, this ignores ALL nonbonded interactions for
    # bonded atoms (i.e., 1-5, 1-6, etc.)
    I = jnp.eye(U_coul.shape[0])
    mask = 1 - I[:, :, jnp.newaxis, jnp.newaxis]
    U_coul_inter = U_coul * mask
    U_coul_inter = (
        0.5 * jnp.where(jnp.isfinite(U_coul_inter), U_coul_inter, 0).sum()
    )  # might work in jax

    return ONE_4PI_EPS0 * (U_coul_inter + U_coul_intra)


# @jit
def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=None):
    """
    calculates total induction energy,
    U_ind = Uself + Uuu + Ustat.

    Arguments:
    <np.array> r
        array of positions for all core and shell sites
    <np.array> q
        array of charges for all core and shell sites
    <np is apparently a <class 'jax._src.interpreters.ad.JVPTracer'> whereas Rij is <class 'jaxlib.xla_extension.ArrayImpl'>.
        array of displacements between core and shell sites
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Uind
        induction energy
    """
    if reshape:
        Dij = jnp.reshape(
            Dij, reshape
        )  # specifically to resolve scipy.optimize handling of 1D arrays

    U_self = Uself(Dij, k)
    U_coul = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    U_coul_static = Ucoul_static(Rij, Qi_shell, Qj_shell, Qi_core, Qj_core)
    U_ind = U_coul - U_coul_static + U_self
    return U_self + U_coul


# @jit
def drudeOpt(
    Rij,
    Dij0,
    Qi_shell,
    Qj_shell,
    Qi_core,
    Qj_core,
    u_scale,
    k,
    methods=["BFGS"],
    d_ref=None,
    reshape=None,
):
    """
    Iteratively determine core/shell displacements, d, by minimizing
    Uind w.r.t d.

    """

    def Uind_min(Dij): return Uind(
        Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape
    )

    for method in methods:
        start = time.time()
        solver = BFGS(fun=Uind_min, tol=0.0001, verbose=False)
        res = solver.run(init_params=Dij0)
        end = time.time()
        d_opt = jnp.reshape(res.params, reshape)
        try:
            if d_ref.any():
                diff = jnp.linalg.norm(d_ref - d_opt)
        except AttributeError:
            pass
    return d_opt


def openmm_inputs_polarization_energy(
    pdb_file,
    xml_file,
    residue_file,
):
    Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, Uind_openmm = (
        openmm_utils.get_inputs(scf=None, pdb_file=pdb_file, xml_file=xml_file, residue_file=residue_file)
    )
    Uind_openmm = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)
    return Uind_openmm


def polarization_energy(R_core, Z_core, atom_types):
    # TODO: assign atom_types here, form R=r_core (NxM_{molecule}x3), r_shell starts as all heavy
    # atoms from R, NxM_{molecules}x3, positions zero for hydrogens
    print('\npolarization energy:')
    print(R_core.shape)
    print(R_core)
    print(Z_core)
    print(atom_types)
    return -0.0105



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
    Every crystalatte energy function plugin must accept the above arguments.

    Takes the `nmers` dictionary; `keynmer`, the key of a given N-mer of
    the N-mers dictionary;

    Results are stored in the `nmer` dictionary under the key `nambe` standing
    for non-additive many-body energy.

    kwargs passed to crystalatte.main() are passed to the energy function
    allowing the user to specify any additional arguments.
    """
    pdb_file = kwargs.get("pdb_file", None)
    xml_file = kwargs.get("xml_file", None)
    atom_types = kwargs.get("atom_types", None)
    residue_file = kwargs.get("residue_file", None)
    print(qcel_mol)
    n_body_energy = -0.0105 
    # update xyz coordinates with qcel_mol.geometry
    atom_types_monomer = [[atom_types]]
    atom_types_dimer = [[atom_types], [atom_types]]
    atom_types_trimer = [[atom_types], [atom_types], [atom_types]]

    atomic_numbers_monomer = [qcel_mol.get_fragment(0).atomic_numbers]
    atomic_numbers_dimer = [atomic_numbers_monomer, atomic_numbers_monomer]
    atomic_numbers_trimer = [atomic_numbers_monomer, atomic_numbers_monomer, atomic_numbers_monomer]


    if len(nmer["monomers"]) == 3:
        # Trimers: ΔE(3)ijk = Eijk − (ΔEij + ΔEik + ΔEjk) − (Ei + Ej + Ek)
        m1, m2, m3 = qcel_mol.get_fragment(0), qcel_mol.get_fragment(1), qcel_mol.get_fragment(2)
        v = np.reshape(m3.geometry, (-1, 1, 3))
        r1 = np.reshape(m1.geometry, (-1, 1, 3))
        r2 = np.reshape(m2.geometry, (-1, 1, 3))
        r3 = np.reshape(m3.geometry, (-1, 1, 3))
        Ei = polarization_energy(r1, atomic_numbers_monomer, atom_types_monomer)
        Ej = polarization_energy(r1, atomic_numbers_monomer, atom_types_monomer)
        Ek = polarization_energy(r1, atomic_numbers_monomer, atom_types_monomer)
        Eij = polarization_energy(np.hstack((r1, r2)),  atomic_numbers_dimer, atom_types_dimer) - Ei - Ej
        Eik = polarization_energy(np.hstack((r1, r3)),  atomic_numbers_dimer, atom_types_dimer) - Ei - Ek
        Ejk = polarization_energy(np.hstack((r2, r3)),  atomic_numbers_dimer, atom_types_dimer) - Ej - Ek
        Eijk = polarization_energy(np.hstack((r1, r2, r3)), atomic_numbers_trimer, atom_types_trimer) - (Eij + Eik + Ejk) - (Ei + Ej + Ek)
        nmer['nambe'] = Eijk
    elif len(nmer["monomers"]) == 2:
        m1, m2 = qcel_mol.get_fragment(0), qcel_mol.get_fragment(1)
        r1 = np.reshape(m1.geometry, (-1, 1, 3))
        r2 = np.reshape(m2.geometry, (-1, 1, 3))
        Ei = polarization_energy(np.reshape(r1, (-1, 1, 3)), atomic_numbers_monomer, atom_types_monomer)
        Ej = polarization_energy(np.reshape(r2, (-1, 1, 3)), atomic_numbers_monomer, atom_types_monomer)
        Eij = polarization_energy(np.hstack((r1, r2)), atomic_numbers_dimer, atom_types_dimer) - Ei - Ej
        nmer['nambe'] = Eij
        # Dimers: ΔE(2)ij = Eij − Ei − Ej
        # m1, m2 = qcel_mol.get_fragment(0), qcel_mol.get_fragment(1)
        # Ei = m1.nuclear_repulsion_energy()
        # Ej = m2.nuclear_repulsion_energy()
        # print(qcel_mol.get_fragment(0).atomic_numbers)
        # print(qcel_mol.get_fragment([0, 1]).atomic_numbers)
        # Eij = qcel_mol.get_fragment([0, 1]).nuclear_repulsion_energy()
        # RA1, RA2 = m1.geometry, m2.geometry
        # ZA1, ZA2 = m1.atomic_numbers, m2.atomic_numbers
        # polarization_energy = openmm_inputs_polarization_energy(
        #     pdb_file=pdb_file,
        #     xml_file=xml_file,
        #     residue_file=residue_file,
        # )
        # polarization_energy /= 2625.5 # convert from kJ/mol to Hartree
        # nmer["nambe"] = polarization_energy
    else:
        raise ValueError("N-mer size not supported")
    return


# U_Pol END

def example_energy_function(
    qcel_mol: qcel.models.Molecule,
    cif_output: str,
    nmers: dict,
    keynmer: str,
    nmer: dict,
    rminseps: str,
    rcomseps: str,
    cle_run_type: list,
    method="method_name_if_applicable",
    bsse_type=None,
    job_memory=None,
    verbose=0,
    **kwargs,
):
    """
    Every crystalatte energy function plugin must accept the above arguments.

    Takes the `nmers` dictionary; `keynmer`, the key of a given N-mer of
    the N-mers dictionary;

    Results are stored in the `nmer` dictionary under the key `nambe` standing
    for non-additive many-body energy.

    kwargs passed to crystalatte.main() are passed to the energy function
    allowing the user to specify any additional arguments.
    """
    example_arg = kwargs.get("example_extra_arg", 0.0)
    print(f"Example extra argument: {example_arg}")
    print(qcel_mol)
    n_body_energy = -0.0105 * np.random.rand()
    if len(nmer["monomers"]) > 2:
        n_minus_1_body_energy = -0.0005
        nmer["nambe"] = n_body_energy - n_minus_1_body_energy

    else:
        nmer["nambe"] = n_body_energy
    return


def main():
    example_energy_function()
    return


if __name__ == "__main__":
    main()
