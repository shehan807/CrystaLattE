import numpy as np
import qcelemental as qcel

def U_ind(
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
    Calculates the induction energy, U_ind, based on the molecular mechanics
    functional form of a polarizable force field, 

    U_ind = U_pol + U_coul, 

    where 
    
    U_pol = 1/2 Σ k_i * ||d_mag_i||^2, 

    and U_coul is the intermolecular coulomb interaction energy + screened (via 
    Thole screening) intramolecular induced-dipole/induced-dipole interaction energy. 
    
    TO-DO: Obtain jnp matrices for...
    ---------------------------------
    positions, Rij 
    core-shell atom displacements, Dij 
    core and shell charges, Q
    thole screening scaling matrix, u_scale 
    core-shell pair harmonic spring constants, k

    """
    example_arg = kwargs.get("example_extra_arg", 0.0)
    print(f"Example extra argument: {example_arg}")
    print(qcel_mol)
    n_body_energy = -0.0105 * np.random.rand()
    
    Dij = drudeOpt(Rij, jnp.ravel(Dij), Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=Dij.shape)
    U_ind = Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k)


    if len(nmer["monomers"]) > 2:
        n_minus_1_body_energy = -0.0005
        nmer["nambe"] = n_body_energy - n_minus_1_body_energy

    else:
        nmer["nambe"] = n_body_energy
    return

# @jit
def Upol(Dij, k):
    """
    Calculates polarization energy, 
    U_pol = 1/2 Σ k_i * ||d_mag_i||^2.

    Arguments:
    <np.array> Dij
        array of displacements between core and shell sites
    <np.array> k
        array of harmonic spring constants for core/shell pairs

    Returns:
    <np.float> Upol
        polarization energy
    """
    d_mag = safe_norm(Dij, 0.0, axis=2)
    return 0.5 * jnp.sum(k * d_mag**2)

# @jit
def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    
    Di = Dij[:,jnp.newaxis,:,jnp.newaxis,:]
    Dj = Dij[jnp.newaxis,:,jnp.newaxis,:,:]

    # use where solution to enable nan-friendly gradients
    Rij_norm       = safe_norm(Rij      , 0.0, axis=-1) 
    Rij_Di_norm    = safe_norm(Rij+Di   , 0.0, axis=-1)
    Rij_Dj_norm    = safe_norm(Rij-Dj   , 0.0, axis=-1)
    Rij_Di_Dj_norm = safe_norm(Rij+Di-Dj, 0.0, axis=-1)
    
    # allow divide by zero 
    _Rij_norm       = jnp.where(Rij_norm == 0.0,       jnp.inf, Rij_norm)  
    _Rij_Di_norm    = jnp.where(Rij_Di_norm == 0.0,    jnp.inf, Rij_Di_norm)
    _Rij_Dj_norm    = jnp.where(Rij_Dj_norm == 0.0,    jnp.inf, Rij_Dj_norm)
    _Rij_Di_Dj_norm = jnp.where(Rij_Di_Dj_norm == 0.0, jnp.inf, Rij_Di_Dj_norm)
    
    #Rij_norm = safe_norm(Rij, 0.0, axis=-1)
    Sij = 1. - (1. + 0.5*Rij_norm*u_scale) * jnp.exp(-u_scale * Rij_norm )
    Sij_Di = 1. - (1. + 0.5*Rij_Di_norm*u_scale) * jnp.exp(-u_scale * Rij_Di_norm )
    Sij_Dj = 1. - (1. + 0.5*Rij_Dj_norm*u_scale) * jnp.exp(-u_scale * Rij_Dj_norm )
    Sij_Di_Dj = 1. - (1. + 0.5*Rij_Di_Dj_norm*u_scale) * jnp.exp(-u_scale * Rij_Di_Dj_norm )
   
    U_coul = Qi_core  * Qj_core  / _Rij_norm\
           + Qi_shell * Qj_core  / _Rij_Di_norm\
           + Qi_core  * Qj_shell / _Rij_Dj_norm\
           + Qi_shell * Qj_shell / _Rij_Di_Dj_norm
    
    # trying with safe_norm 
    U_coul_intra = Sij       * -Qi_shell * -Qj_shell  / _Rij_norm\
                 + Sij_Di    *  Qi_shell * -Qj_shell  / _Rij_Di_norm\
                 + Sij_Dj    * -Qi_shell *  Qj_shell  / _Rij_Dj_norm\
                 + Sij_Di_Dj *  Qi_shell *  Qj_shell  / _Rij_Di_Dj_norm
    
    # keep diagonal (intramolecular) components except for self-terms
    I_intra = jnp.eye(U_coul_intra.shape[0])
    I_self  = jnp.eye(U_coul_intra.shape[-1])
    U_coul_intra = (U_coul_intra * I_intra[:,:,jnp.newaxis,jnp.newaxis]) * (1 - I_self[jnp.newaxis,jnp.newaxis,:,:])
    U_coul_intra = 0.5 * jnp.where(jnp.isfinite(U_coul_intra), U_coul_intra, 0).sum() # might work in jax
    
    # remove diagonal (intramolecular) components
    I = jnp.eye(U_coul.shape[0])
    U_coul_inter = U_coul * (1 - I[:,:,jnp.newaxis,jnp.newaxis])
    
    U_coul_inter = 0.5 * jnp.where(jnp.isfinite(U_coul_inter), U_coul_inter, 0).sum() # might work in jax
    return ONE_4PI_EPS0*(U_coul_inter + U_coul_intra)
def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=None):
    """
    calculates total induction energy, 
    U_ind = Upol + Uuu + Ustat.

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
        Dij = jnp.reshape(Dij,reshape) # specifically to resolve scipy.optimize handling of 1D arrays

    U_pol  = Upol(Dij, k)
    U_coul = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    logger.debug(f"U_pol = {U_pol} kJ/mol\nU_coul = {U_coul}\n")
    
    return U_pol + U_coul
def drudeOpt(Rij, Dij0, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, methods=["BFGS"],d_ref=None, reshape=None):
    """
    Iteratively determine core/shell displacements, d, by minimizing 
    Uind w.r.t d. 

    """
    from jaxopt import BFGS, LBFGS, ScipyMinimize
    Uind_min = lambda Dij: Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape)
     
    for method in methods:
        start = time.time()
        solver = BFGS(fun=Uind_min)
        res = solver.run(init_params=Dij0)
        end = time.time()
        logger.info(f"JAXOPT.BFGS Minimizer completed in {end-start:.3f} seconds!!")
        d_opt = jnp.reshape(res.params,reshape)
        try:
            if d_ref.any():
                diff = jnp.linalg.norm(d_ref-d_opt)
        except AttributeError:
            pass
    return d_opt

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

    Questions: Why are there no charge or multiplictity arguments anywhere in the code?
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
