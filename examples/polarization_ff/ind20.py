"""
Module for calculating induction energy using Fortran fitting code.
This module provides a clean interface to run the Fortran executable
and extract the fitted induction energy values.
"""

import os
import subprocess
import tempfile
import re
import numpy as np


# Conversion factor from Angstrom to Bohr
ANG_TO_BOHR = 1.8897259886


def qcel_to_bohr_format(qcel_mol, output_file):
    """
    Convert a QCElemental molecule object to the bohr format required by the Fortran code.
    
    Args:
        qcel_mol: QCElemental molecule object containing dimer coordinates
        output_file: Path to write the bohr format file
    """
    # Extract coordinates and symbols
    coords = qcel_mol.geometry  # Flat array in bohr units already
    symbols = qcel_mol.symbols
    
    # QCElemental stores geometry in bohr by default, so no conversion needed
    coords_bohr = coords.reshape(-1, 3) 
    
    # Assume the dimer is composed of two identical molecules
    # Split the coordinates in half
    n_atoms_total = len(symbols)
    n_atoms_per_monomer = n_atoms_total // 2
    
    with open(output_file, 'w') as f:
        # Write first monomer
        f.write(f"{n_atoms_per_monomer}\n")
        for i in range(n_atoms_per_monomer):
            f.write(f" {symbols[i]:2s}    {coords_bohr[i,0]:16.6f} {coords_bohr[i,1]:16.6f} {coords_bohr[i,2]:16.6f}\n")
        
        # Write second monomer
        f.write(f"{n_atoms_per_monomer}\n")
        for i in range(n_atoms_per_monomer, n_atoms_total):
            f.write(f" {symbols[i]:2s}    {coords_bohr[i,0]:16.6f} {coords_bohr[i,1]:16.6f} {coords_bohr[i,2]:16.6f}\n")
        
        # Add dummy energy section (required by format but not used in calculation)
        # Match exact format from working imidazole_tmp.bohr file
        f.write(" E1pol                   0.00000000\n")
        f.write(" E1exch                  0.00000000\n")
        f.write(" E1exch(S2)              0.00000000\n")
        f.write(" E2ind(unc)              0.00000000\n")
        f.write(" E2ind                   0.00000000\n")
        f.write(" E2ind-exch              0.00000000\n")
        f.write(" E2disp(unc)             0.00000000\n")
        f.write(" E2disp                  0.00000000\n")
        f.write(" E2disp-exch(unc)        0.00000000\n")
        f.write(" E2disp-exch             0.00000000\n")
        f.write(" E1tot                   0.00000000\n")
        f.write(" E2tot                   0.00000000\n")
        f.write(" E1tot+E2tot             0.00000000\n")
        f.write(" E2ind[B->A]             0.00000000\n")
        f.write(" E2ind[A->B]             0.00000000\n")
        f.write(" E2exchind_BA            0.00000000\n")
        f.write(" E2exchind_AB            0.00000000\n")
        f.write(" dhf                     0.00000000\n")


def extract_energy_components_from_output(output_file):
    """
    Extract the FIT energy values (induction, electrostatics, dhf) from the Fortran output file.
    
    Args:
        output_file: Path to the Fortran output file
        
    Returns:
        dict: Dictionary containing fitted energy values in kJ/mol
              Keys: 'ind20_fit', 'elec_fit', 'dhf_fit'
    """
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Look for the patterns for each energy component
    # The patterns in the output are:
    # Exch energy: SAPT , FIT
    # Elec energy: SAPT , FIT  
    # Induc energy: SAPT , FIT
    # dhf energy: SAPT , FIT
    
    lines = content.split('\n')
    energy_components = {}
    
    for i, line in enumerate(lines):
        if "Induc energy: SAPT , FIT" in line:
            # The values are on the next line
            if i + 1 < len(lines):
                values_line = lines[i + 1]
                values = values_line.strip().split()
                if len(values) >= 2:
                    fit_value = float(values[1])
                    # Convert from Hartree to kJ/mol
                    energy_components['ind20_fit'] = fit_value * 2625.4996
        
        elif "Elec energy: SAPT , FIT" in line:
            if i + 1 < len(lines):
                values_line = lines[i + 1]
                values = values_line.strip().split()
                if len(values) >= 2:
                    fit_value = float(values[1])
                    energy_components['elec_fit'] = fit_value * 2625.4996
        
        elif "dhf energy: SAPT , FIT" in line:
            if i + 1 < len(lines):
                values_line = lines[i + 1]
                values = values_line.strip().split()
                if len(values) >= 2:
                    fit_value = float(values[1])
                    energy_components['dhf_fit'] = fit_value * 2625.4996
    
    # Check that we found all expected components
    expected_keys = ['ind20_fit', 'elec_fit', 'dhf_fit']
    missing_keys = [key for key in expected_keys if key not in energy_components]
    if missing_keys:
        raise ValueError(f"Could not find energy components: {missing_keys} in output file")
    print(energy_components) 
    return energy_components


def extract_ind20_from_output(output_file):
    """
    Extract the FIT induction energy value from the Fortran output file.
    (Kept for backward compatibility)
    
    Args:
        output_file: Path to the Fortran output file
        
    Returns:
        float: The fitted induction energy in kJ/mol
    """
    energy_components = extract_energy_components_from_output(output_file)
    return energy_components['ind20_fit']


def get_ind20_fit(qcel_mol, molecule_name, fortran_dir=None):
    """
    Calculate the fitted induction energy (ind20) using the Fortran code.
    
    Args:
        qcel_mol: QCElemental molecule object
        molecule_name: Name of the molecule (e.g., 'pyrazole', 'imidazole')
        fortran_dir: Directory containing the Fortran executable (optional)
        
    Returns:
        float: The fitted induction energy in kJ/mol
    """
    # Use the molecules directory for all operations
    molecules_dir = os.path.join(
        os.path.dirname(__file__),
        "ind20_ff"
    )
    
    # Paths to required files
    executable = os.path.join(molecules_dir, "fitenergy")
    param_file = os.path.join(molecules_dir, f"{molecule_name}_param")
    
    # Copy fitenergy executable from single_dimer to molecules directory if needed
    if not os.path.exists(executable):
        single_dimer_executable = os.path.join(
            os.path.dirname(__file__),
            "molecules/fitenergy"
        )
        if os.path.exists(single_dimer_executable):
            import shutil
            shutil.copy2(single_dimer_executable, executable)
        else:
            raise FileNotFoundError(f"Fortran executable not found: {single_dimer_executable}")
    
    # Check if parameter file exists
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    
    # Create input file in bohr format in molecules directory
    input_file = os.path.join(molecules_dir, f"{molecule_name}_tmp.bohr")
    qcel_to_bohr_format(qcel_mol, input_file)
    
    # Output file path in molecules directory
    output_file = os.path.join(molecules_dir, "testout_tmp")
    
    # Run the Fortran executable from molecules directory
    cmd = f"rm testout_tmp; ./fitenergy {molecule_name}_tmp.bohr {molecule_name}_param testout_tmp"
    
    print(cmd, molecules_dir, molecule_name)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
        executable='/bin/zsh',
        check=False,  # Don't raise exception on non-zero exit
        cwd=molecules_dir
    )
    
    # Check if output file was created (ignore exit code)
    if not os.path.exists(output_file):
        raise RuntimeError(f"Fortran execution failed - no output file created")
    
    # Extract the ind20 value from the output
    ind20_value = extract_ind20_from_output(output_file)
    
    return ind20_value


def get_all_fit_energies(qcel_mol, molecule_name, fortran_dir=None):
    """
    Calculate all fitted energy components (ind20, elec, dhf) using the Fortran code.
    
    Args:
        qcel_mol: QCElemental molecule object
        molecule_name: Name of the molecule (e.g., 'pyrazole', 'imidazole')
        fortran_dir: Directory containing the Fortran executable (optional)
        
    Returns:
        dict: Dictionary containing fitted energy values in kJ/mol
              Keys: 'ind20_fit', 'elec_fit', 'dhf_fit'
    """
    # Use the molecules directory for all operations
    molecules_dir = os.path.join(
        os.path.dirname(__file__),
        "molecules"
    )
    
    # Paths to required files
    executable = os.path.join(molecules_dir, "fitenergy")
    param_file = os.path.join(molecules_dir, f"{molecule_name}_param")
    
    # Check if parameter file exists
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    
    # Create input file in bohr format in molecules directory
    input_file = os.path.join(molecules_dir, f"{molecule_name}_tmp.bohr")
    qcel_to_bohr_format(qcel_mol, input_file)
    
    # Output file path in molecules directory
    output_file = os.path.join(molecules_dir, "testout_tmp")
    
    # Remove output file if it exists (Fortran code fails if file already exists)
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Run the Fortran executable from molecules directory
    cmd = f"./fitenergy {molecule_name}_tmp.bohr {molecule_name}_param testout_tmp"
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
        executable='/bin/zsh',
        check=False,  # Don't raise exception on non-zero exit
        cwd=molecules_dir
    )
    
    # Check if output file was created (ignore exit code)
    if not os.path.exists(output_file):
        raise RuntimeError(f"Fortran execution failed - no output file created")
    
    # Extract all energy components immediately after creation
    energy_components = extract_energy_components_from_output(output_file)
    
    return energy_components


# Optional: Add a batch processing function for efficiency
def get_ind20_fit_batch(qcel_mols, molecule_name, fortran_dir=None):
    """
    Process multiple QCElemental molecules in batch.
    
    Args:
        qcel_mols: List of QCElemental molecule objects
        molecule_name: Name of the molecule
        fortran_dir: Directory containing the Fortran executable
        
    Returns:
        list: Fitted induction energies in kJ/mol
    """
    results = []
    for qcel_mol in qcel_mols:
        try:
            ind20 = get_ind20_fit(qcel_mol, molecule_name, fortran_dir)
            results.append(ind20)
        except Exception as e:
            print(f"Warning: Failed to calculate ind20 for a configuration: {e}")
            results.append(np.nan)
    
    return results
