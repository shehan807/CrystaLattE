#!/usr/bin/env python
"""
Optimize polarizability parameters for acetic acid force field to maximize R^2
between MD and SAPT induction energies.
"""

import os
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import subprocess
import time
from scipy.optimize import minimize
import shutil

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Install with 'pip install tqdm' for progress bars")
    # Simple fallback
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_description(self, desc):
            print(f"\r{desc}", end="")
        def set_postfix(self, postfix):
            pass
        def write(self, msg):
            print(msg)
        def close(self):
            print()  # New line

# Try to import jaxopt, fallback to scipy if not available
try:
    import jaxopt
    import jax.numpy as jnp
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False
    print("JAXopt not found, using scipy.optimize instead")


class PolarizabilityOptimizer:
    def __init__(self, molecule="acetic_acid", distance_cutoff=12):
        self.molecule = molecule
        self.distance_cutoff = distance_cutoff
        self.base_xml = f"{molecule}/{molecule}.xml"
        self.temp_xml = f"{molecule}/{molecule}_tmp.xml"
        self.output_csv = f"{molecule}/{molecule}.csv"
        self.optimization_log = f"{molecule}/{molecule}_optimization_log.csv"
        
        # Parse base polarizabilities, charges, and Thole parameters from XML
        self.base_polarizabilities = self.parse_polarizabilities(self.base_xml)
        self.charges = self.parse_charges(self.base_xml)
        self.base_thole = self.parse_thole(self.base_xml)
        self.n_particles = len(self.base_polarizabilities)
        self.n_params = 2 * self.n_particles  # Both polarizability and Thole for each particle
        
        # Set bounds: ±100% for polarizabilities, ±20% for Thole
        self.bounds = []
        for i in range(self.n_particles):
            # Polarizability bounds: 0 to 2x (±100%)
            self.bounds.append((0.0, 2.0 * self.base_polarizabilities[i]))
            # Thole bounds: ±20%
            self.bounds.append((0.8 * self.base_thole[i], 1.2 * self.base_thole[i]))
        
        # Track optimization progress
        self.iteration = 0
        self.best_r2 = -np.inf
        self.best_params = None
        self.history = []
        self.pbar = None  # Progress bar
        
    def parse_polarizabilities(self, xml_file):
        """Extract polarizability values from XML file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        polarizabilities = []
        for particle in root.find('DrudeForce').findall('Particle'):
            pol = float(particle.get('polarizability'))
            polarizabilities.append(pol)
            
        return polarizabilities
    
    def parse_charges(self, xml_file):
        """Extract Drude charge values from XML file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        charges = []
        for particle in root.find('DrudeForce').findall('Particle'):
            charge = float(particle.get('charge'))
            charges.append(charge)
            
        return charges
    
    def parse_thole(self, xml_file):
        """Extract Thole parameter values from XML file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        thole_params = []
        for particle in root.find('DrudeForce').findall('Particle'):
            thole = float(particle.get('thole'))
            thole_params.append(thole)
            
        return thole_params
    
    def update_xml_parameters(self, params):
        """Create temporary XML file with updated polarizability and Thole values."""
        # Extract polarizabilities and Thole parameters from flat array
        polarizabilities = []
        thole_params = []
        for i in range(self.n_particles):
            polarizabilities.append(params[2*i])
            thole_params.append(params[2*i + 1])
        
        # Copy original file
        shutil.copy(self.base_xml, self.temp_xml)
        
        # Parse and update
        tree = ET.parse(self.temp_xml)
        root = tree.getroot()
        
        particles = root.find('DrudeForce').findall('Particle')
        for i, particle in enumerate(particles):
            particle.set('polarizability', f"{polarizabilities[i]:.8f}")
            particle.set('thole', f"{thole_params[i]:.8f}")
        
        # Write updated XML
        tree.write(self.temp_xml, encoding='unicode', xml_declaration=True)
        
    def run_benchmark(self):
        """Run U_ind_benchmark.py for the molecule with temporary XML."""
        env = os.environ.copy()
        env['MOLECULES_LIST'] = self.molecule
        
        # Update progress bar description
        if self.pbar:
            self.pbar.set_description(f"Running benchmark for iteration {self.iteration}")
        
        # We need to temporarily modify the benchmark to use our temp XML
        # For now, we'll create a modified version
        self.create_modified_benchmark()
        
        try:
            result = subprocess.run(
                [sys.executable, "U_ind_benchmark_temp.py"],
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            if result.returncode != 0:
                print(f"Benchmark error: {result.stderr}")
                return False
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed: {e}")
            return False
        finally:
            # Cleanup temp benchmark script
            if os.path.exists("U_ind_benchmark_temp.py"):
                os.remove("U_ind_benchmark_temp.py")
        
        return True
    
    def create_modified_benchmark(self):
        """Create a temporary version of benchmark script that uses temp XML."""
        with open("U_ind_benchmark.py", 'r') as f:
            content = f.read()
        
        # Modify the ff_file line to use our temp XML
        content = content.replace(
            'ff_file        = os.path.join(molecule, molecule + ".xml")',
            f'ff_file        = os.path.join(molecule, molecule + "_tmp.xml")'
        )
        
        with open("U_ind_benchmark_temp.py", 'w') as f:
            f.write(content)
    
    def calculate_r2(self):
        """Calculate R^2 from the benchmark results CSV."""
        try:
            df = pd.read_csv(self.output_csv)
            
            # Apply distance cutoff if specified
            if self.distance_cutoff is not None:
                df_filtered = df[df["distance"] > self.distance_cutoff]
                if self.pbar:
                    self.pbar.set_postfix({
                        'cutoff': f'{self.distance_cutoff}Å',
                        'points': f'{len(df_filtered)}/{len(df)}'
                    })
                df = df_filtered
            
            # Extract induction energies
            x_sapt = df["Uind_sapt"].values
            y_md = df["Uind_md"].values
            
            # Calculate R^2
            r2 = np.corrcoef(x_sapt, y_md)[0, 1] ** 2
            
            # Also calculate other metrics for logging
            errors = y_md - x_sapt
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            
            return r2, mae, rmse
            
        except Exception as e:
            print(f"Error calculating R^2: {e}")
            return 0.0, np.inf, np.inf
    
    def calculate_k_constants(self, polarizabilities):
        """Calculate k constants in atomic units from charge and polarizability.
        
        k = q^2 / alpha
        
        where:
        - q is in elementary charges (e)
        - alpha is in nm^3
        - k is returned in atomic units (Hartree/Bohr^2)
        
        Conversion factors:
        - 1 nm = 18.89726 Bohr
        - 1 nm^3 = 6748.334 Bohr^3
        """
        # Conversion factor from nm^3 to Bohr^3
        nm3_to_bohr3 = 6748.334
        
        k_values = []
        for charge, alpha in zip(self.charges, polarizabilities):
            # q^2 in e^2
            q_squared = charge ** 2
            
            # Convert alpha from nm^3 to Bohr^3
            alpha_bohr3 = alpha * nm3_to_bohr3
            
            # Calculate k in atomic units (Hartree/Bohr^2)
            k = q_squared / alpha_bohr3
            k_values.append(k)
            
        return k_values
    
    def objective(self, params):
        """Objective function: negative R^2 (for minimization)."""
        self.iteration += 1
        
        # Update progress bar
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_description(f"Iteration {self.iteration}")
        
        # Update XML with new parameters (polarizabilities and Thole)
        self.update_xml_parameters(params)
        
        # Run benchmark
        success = self.run_benchmark()
        if not success:
            if self.pbar:
                self.pbar.write("Benchmark failed, returning penalty")
            return 1e6  # Large penalty for failed runs
        
        # Calculate R^2
        r2, mae, rmse = self.calculate_r2()
        
        # Extract polarizabilities for logging
        polarizabilities = [params[2*i] for i in range(self.n_particles)]
        thole_params = [params[2*i + 1] for i in range(self.n_particles)]
        
        # Log progress (simplified for variable number of particles)
        log_entry = {
            'iteration': self.iteration,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
        }
        for i in range(self.n_particles):
            log_entry[f'pol_{i}'] = polarizabilities[i]
            log_entry[f'thole_{i}'] = thole_params[i]
        self.history.append(log_entry)
        
        # Update progress bar with current metrics
        if self.pbar:
            self.pbar.set_postfix({
                'R²': f'{r2:.6f}',
                'Best R²': f'{self.best_r2:.6f}',
                'MAE': f'{mae:.2e}'
            })
        
        # Track best result
        if r2 > self.best_r2:
            self.best_r2 = r2
            self.best_params = params.copy()
            if self.pbar:
                self.pbar.write(f"✓ New best R²: {r2:.6f}")
        
        # Save progress
        pd.DataFrame(self.history).to_csv(self.optimization_log, index=False)
        
        return -r2  # Negative for minimization
    
    def optimize(self, method='L-BFGS-B', maxiter=50):
        """Run the optimization."""
        print(f"Starting optimization of {self.molecule} polarizabilities and Thole parameters")
        print(f"Number of particles: {self.n_particles}")
        print(f"Total parameters: {self.n_params} ({self.n_particles} polarizabilities + {self.n_particles} Thole)")
        print(f"Base polarizabilities: {self.base_polarizabilities}")
        print(f"Base Thole parameters: {self.base_thole}")
        print(f"Polarizability bounds: 0 to 2x (±100%)")
        print(f"Thole bounds: 0.8x to 1.2x (±20%)")
        print(f"Distance cutoff: {self.distance_cutoff} Å")
        print(f"Optimization method: {method}")
        print(f"Max iterations: {maxiter}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Initialize progress bar
        self.pbar = tqdm(total=maxiter, desc="Optimizing", unit="iter")
        
        # Create initial parameter array (alternating polarizabilities and Thole)
        init_params = []
        for i in range(self.n_particles):
            init_params.append(self.base_polarizabilities[i])
            init_params.append(self.base_thole[i])
        
        try:
            if HAS_JAXOPT and method == 'jaxopt':
                # Use JAXopt bounded optimizer
                print("Using JAXopt optimizer")
                # Convert to JAX arrays
                init_params_jax = jnp.array(init_params)
                lower_bounds = jnp.array([b[0] for b in self.bounds])
                upper_bounds = jnp.array([b[1] for b in self.bounds])
                
                # Use ScipyBoundedMinimize wrapper
                optimizer = jaxopt.ScipyBoundedMinimize(
                    fun=self.objective,
                    method='L-BFGS-B'
                )
                
                result = optimizer.run(
                    init_params=init_params_jax,
                    bounds=(lower_bounds, upper_bounds)
                )
                optimal_params = result.params
                
            else:
                # Use scipy optimizer
                result = minimize(
                    fun=self.objective,
                    x0=init_params,
                    method=method,
                    bounds=self.bounds,
                    options={'maxiter': maxiter, 'disp': False}  # Turn off scipy's display
                )
                optimal_params = result.x
        finally:
            # Close progress bar
            if self.pbar:
                self.pbar.close()
        
        end_time = time.time()
        
        # Final results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Total iterations: {self.iteration}")
        print(f"Best R^2: {self.best_r2:.6f}")
        
        # Extract optimized polarizabilities and Thole parameters
        opt_polarizabilities = [self.best_params[2*i] for i in range(self.n_particles)]
        opt_thole = [self.best_params[2*i + 1] for i in range(self.n_particles)]
        
        # Calculate k constants for base and optimized parameters
        base_k_values = self.calculate_k_constants(self.base_polarizabilities)
        opt_k_values = self.calculate_k_constants(opt_polarizabilities)
        
        print(f"\nOptimal parameters:")
        print(f"{'Particle':<12} {'Charge (e)':<12} {'α_base (nm³)':<12} {'α_opt (nm³)':<12} {'α Change':<10} {'Thole_base':<12} {'Thole_opt':<12} {'T Change':<10}")
        print("-" * 112)
        
        # Dynamically set particle types based on molecule
        if self.molecule == 'acetic_acid':
            particle_types = ['ACA-DC00', 'ACA-DC01', 'ACA-DO02', 'ACA-DO03']
        elif self.molecule == 'pyrazole':
            particle_types = ['PYR-DN00', 'PYR-DN01', 'PYR-DC02', 'PYR-DC03', 'PYR-DC04']
        elif self.molecule == 'pyrazine':
            particle_types = ['PYR-DN00', 'PYR-DN04', 'PYR-DC01', 'PYR-DC02', 'PYR-DC03', 'PYR-DC05']
        elif self.molecule == 'imidazole':
            particle_types = ['IMI-DN00', 'IMI-DC01', 'IMI-DN02', 'IMI-DC03', 'IMI-DC04']  # Update if needed
        else:
            # Generic naming if molecule not recognized
            particle_types = [f'Particle_{i}' for i in range(self.n_particles)]
        for i in range(self.n_particles):
            pol_change = (opt_polarizabilities[i] - self.base_polarizabilities[i]) / self.base_polarizabilities[i] * 100
            thole_change = (opt_thole[i] - self.base_thole[i]) / self.base_thole[i] * 100
            print(f"{particle_types[i]:<12} {self.charges[i]:<12.6f} {self.base_polarizabilities[i]:<12.8f} {opt_polarizabilities[i]:<12.8f} {pol_change:+9.1f}% {self.base_thole[i]:<12.6f} {opt_thole[i]:<12.6f} {thole_change:+9.1f}%")
        
        print(f"\nk constants (a.u.):")
        print(f"{'Particle':<12} {'k_base':<15} {'k_opt':<15} {'Change':<10}")
        print("-" * 52)
        for i in range(self.n_particles):
            k_change = (opt_k_values[i] - base_k_values[i]) / base_k_values[i] * 100
            print(f"{particle_types[i]:<12} {base_k_values[i]:<15.6f} {opt_k_values[i]:<15.6f} {k_change:+9.1f}%")
        
        # Create final optimized XML
        self.update_xml_parameters(self.best_params)
        optimized_xml = f"{self.molecule}/{self.molecule}_optimized.xml"
        shutil.copy(self.temp_xml, optimized_xml)
        print(f"\nOptimized XML saved to: {optimized_xml}")
        
        # Cleanup
        if os.path.exists(self.temp_xml):
            os.remove(self.temp_xml)
        
        return self.best_params, self.best_r2


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimize polarizability parameters for force field"
    )
    parser.add_argument(
        "--molecule", default="acetic_acid",
        help="Molecule to optimize (default: acetic_acid)"
    )
    parser.add_argument(
        "--distance-cutoff", type=float, default=12,
        help="Distance cutoff in Angstroms (default: 12)"
    )
    parser.add_argument(
        "--method", default="L-BFGS-B",
        help="Optimization method (default: L-BFGS-B)"
    )
    parser.add_argument(
        "--maxiter", type=int, default=50,
        help="Maximum iterations (default: 50)"
    )
    
    args = parser.parse_args()
    
    optimizer = PolarizabilityOptimizer(
        molecule=args.molecule,
        distance_cutoff=args.distance_cutoff
    )
    
    optimal_params, best_r2 = optimizer.optimize(
        method=args.method,
        maxiter=args.maxiter
    )


if __name__ == "__main__":
    main()