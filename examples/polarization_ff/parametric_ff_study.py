#!/usr/bin/env python3
"""
Parametric force field study to evaluate different charge models.
Tests combinations of static charges (ESP, RESP, MPFIT) and drude charges.
Calculates MAE and RMSE for different distance ranges.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from crystalatte.plugins import force_fields
from ff_generator import update_ff
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# Suppress MDAnalysis info messages
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def calculate_metrics(ref_values, calc_values):
    """Calculate MAE, RMSE, and R² for a set of values."""
    # Convert to numpy arrays, forcing numeric type
    ref_values = pd.to_numeric(ref_values, errors='coerce')
    calc_values = pd.to_numeric(calc_values, errors='coerce')
    
    # Convert to numpy arrays if they're pandas Series
    if hasattr(ref_values, 'values'):
        ref_values = ref_values.values
    if hasattr(calc_values, 'values'):
        calc_values = calc_values.values
    
    # Ensure they're numpy arrays of floats
    ref_values = np.array(ref_values, dtype=float)
    calc_values = np.array(calc_values, dtype=float)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(ref_values) | np.isnan(calc_values))
    
    if np.sum(valid_mask) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'max_error': np.nan
        }
    
    ref_clean = ref_values[valid_mask]
    calc_clean = calc_values[valid_mask]
    
    errors = calc_clean - ref_clean
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': np.nan,  # Skip R² calculation for now
        'max_error': np.max(np.abs(errors)) if len(errors) > 0 else np.nan
    }

def create_parity_plot(df, ref_col, calc_col, output_path, 
                       energy_type="induction", distance_range="all",
                       molecule="", model_name="", fontsize=14):
    """
    Create a parity plot comparing reference and calculated values.
    
    Parameters:
    -----------
    df : DataFrame
        Data with reference and calculated values
    ref_col : str
        Column name for reference values
    calc_col : str  
        Column name for calculated values
    output_path : str
        Path to save the plot
    energy_type : str
        Type of energy being plotted (for labels)
    distance_range : str
        Distance range description
    molecule : str
        Molecule name for title
    model_name : str
        Force field model description
    fontsize : int
        Base font size for plot
    """
    # Get values and remove NaN
    ref_values = pd.to_numeric(df[ref_col], errors='coerce')
    calc_values = pd.to_numeric(df[calc_col], errors='coerce')
    distances = df['distance'].values
    
    # Convert to numpy arrays
    ref_values = np.array(ref_values, dtype=float)
    calc_values = np.array(calc_values, dtype=float)
    
    valid_mask = ~(np.isnan(ref_values) | np.isnan(calc_values))
    ref_clean = ref_values[valid_mask]
    calc_clean = calc_values[valid_mask]
    dist_clean = distances[valid_mask]
    
    if len(ref_clean) == 0:
        logger.warning(f"No valid data for plot: {output_path}")
        return
    
    # Calculate metrics
    errors = calc_clean - ref_clean
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Setup plot style
    plt.rcParams.update({
        "font.family": "serif",
        "xtick.labelsize": fontsize * 0.8,
        "ytick.labelsize": fontsize * 0.8,
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Style axes
    border_width = 1.5
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(border_width)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction="in", length=6, width=border_width, 
                   which="major", top=True, right=True)
    ax.tick_params(direction="in", length=3, width=border_width,
                   which="minor", top=True, right=True)
    
    # Labels
    if energy_type.lower() == "electrostatics":
        ax.set_xlabel("SAPT Electrostatics (kJ/mol)", fontsize=fontsize)
        ax.set_ylabel("MD Electrostatics (kJ/mol)", fontsize=fontsize)
    elif energy_type.lower() == "induction":
        ax.set_xlabel("SAPT Induction (kJ/mol)", fontsize=fontsize)
        ax.set_ylabel("MD Induction (kJ/mol)", fontsize=fontsize)
    
    # Title
    title = f"{molecule} - {model_name}\n{energy_type.capitalize()} ({distance_range})"
    ax.set_title(title, fontsize=fontsize)
    
    # Scatter plot colored by distance
    sc = ax.scatter(ref_clean, calc_clean, c=dist_clean, cmap="viridis", 
                   alpha=0.8, s=40, edgecolors="none")
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Distance (Å)", fontsize=fontsize * 0.9)
    
    # Diagonal line and bounds
    min_val = min(ref_clean.min(), calc_clean.min())
    max_val = max(ref_clean.max(), calc_clean.max())
    line_vals = np.linspace(min_val, max_val, 200)
    ax.plot(line_vals, line_vals, "k--", lw=2, alpha=0.7)
    
    # Chemical accuracy band (1 kJ/mol)
    chemical_accuracy = 1.0
    ax.fill_between(line_vals, line_vals - chemical_accuracy, 
                   line_vals + chemical_accuracy,
                   color="orange", alpha=0.2, 
                   label=f"±{chemical_accuracy} kJ/mol")
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # Statistics text
    stat_text = (
        f"MAE: {mae:.4f} kJ/mol\n"
        f"RMSE: {rmse:.4f} kJ/mol\n"
        f"N points: {len(ref_clean)}"
    )
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes,
           fontsize=fontsize * 0.8, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    # Error distribution inset
    left, bottom, width, height = 0.62, 0.15, 0.25, 0.2
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.hist(errors, bins=30, color="steelblue", alpha=0.7, edgecolor="k")
    ax_inset.set_title("Error Distribution", fontsize=fontsize * 0.7)
    ax_inset.tick_params(axis="both", which="major", labelsize=fontsize * 0.6)
    
    # Center error histogram on zero
    half_range = max(abs(errors.min()), abs(errors.max()), chemical_accuracy * 2)
    ax_inset.set_xlim(-half_range, half_range)
    ax_inset.set_yticks([])
    ax_inset.axvline(0.0, color="k", linestyle="--", linewidth=1.5)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.debug(f"Saved plot to {output_path}")

def analyze_distance_ranges(df, ref_col, calc_col, distance_ranges=None):
    """Analyze metrics for different distance ranges."""
    if distance_ranges is None:
        distance_ranges = [
            ('all', 0, 100),
            ('short', 0, 5),
            ('medium', 5, 10),
            ('long', 10, 20)
        ]
    
    results = {}
    for range_name, min_dist, max_dist in distance_ranges:
        df_subset = df[(df['distance'] >= min_dist) & (df['distance'] < max_dist)]
        
        if len(df_subset) > 0:
            metrics = calculate_metrics(
                df_subset[ref_col].values,
                df_subset[calc_col].values
            )
            metrics['n_points'] = len(df_subset)
        else:
            metrics = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 
                      'max_error': np.nan, 'n_points': 0}
        
        results[range_name] = metrics
    
    return results

def run_single_ff_test(molecule, ff_file, pkl_file, pdb_file, atom_types_map):
    """Run force field test for a single molecule and FF combination."""
    
    # Load the data
    df = pd.read_pickle(pkl_file)
    results = []
    
    start_time = time.time()
    
    for index, row in df.iterrows():
        qcel_mol = row["mol"]
        distance = row["Minimum Monomer Separations (A)"]
        
        # Get reference values from SAPT
        Uind_sapt = row.get("SAPT0 Induction (kJ/mol)", np.nan)
        Ues_sapt = row.get("SAPT0 Electrostatics (kJ/mol)", np.nan)
        
        # Calculate MD energies
        Uind_md, Udf, Unb, Ues = force_fields.polarization_energy_sample(
            qcel_mol,
            pdb_file=pdb_file,
            xml_file=ff_file,
            atom_types_map=atom_types_map,
            omm_decomp=True
        )
        # print(f"(Uind_md, Udf, Unb, Ues, distance) = ({Uind_md}, {Udf}, {Unb}, {Unb}, {Ues}, {distance})")
        results.append({
            'distance': distance,
            'nmer_name': row["N-mer Name"],
            'Uind_md': Uind_md,
            'Udf': Udf,
            'Unb': Unb,
            'Ues': Ues,
            'Uind_sapt': Uind_sapt,
            'Ues_sapt': Ues_sapt
        })
    
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df['time_per_system'] = elapsed_time / len(results) if results else 0
    
    return results_df

def run_parametric_study(molecules=None, static_models=None, drude_models=None, 
                         output_dir='parametric_results', create_plots=True):
    """
    Run comprehensive parametric study of force field charge models.
    
    Parameters:
    -----------
    molecules : list or None
        List of molecules to test. If None, uses all available.
    static_models : list or None
        List of static charge models. If None, uses all available.
    drude_models : list or None
        List of drude charge models. If None, uses all available.
    output_dir : str
        Directory to save results.
    """
    
    # Load charge database to get available options
    with open("charge_database/charge_models_v3.json", 'r') as f:
        charge_db = json.load(f)
    
    # Default to all available options if not specified
    if molecules is None:
        molecules = list(charge_db['molecules'].keys())
    if static_models is None:
        static_models = ['ESP', 'RESP', 'MPFIT']
    if drude_models is None:
        drude_models = ['2013JPC']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Main nested loops
    for molecule in molecules:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing molecule: {molecule}")
        logger.info(f"{'='*60}")
        
        # Check what charge models are available for this molecule
        mol_data = charge_db['molecules'].get(molecule, {})
        available_static = [m for m in static_models 
                          if m in mol_data.get('static_charges', {})]
        available_drude = [m for m in drude_models 
                         if m in mol_data.get('drude_charges', {})]
        
        if not available_static:
            logger.warning(f"No static charge models available for {molecule}")
            continue
        
        # Prepare molecule files
        pkl_file = os.path.join(molecule, f"{molecule}.pkl")
        pdb_file = os.path.join(molecule, f"{molecule}.pdb")
        atom_types_map = os.path.join(molecule, f"{molecule}_map.csv")
        residue_file = os.path.join(molecule, f"{molecule}_residue.xml")
        
        if not all(os.path.exists(f) for f in [pkl_file, pdb_file, atom_types_map]):
            logger.warning(f"Missing required files for {molecule}")
            continue
        
        # Loop through charge models
        for static_model in available_static:
            for drude_model in available_drude:
                logger.info(f"\nTesting: {molecule} / {static_model} / {drude_model}")
                
                # Generate or get force field file
                try:
                    ff_file = update_ff(molecule, static_model, drude_model)
                except Exception as e:
                    logger.error(f"Failed to generate FF: {e}")
                    continue
                
                # Run the test
                results_df = run_single_ff_test(
                    molecule, ff_file, pkl_file, pdb_file, atom_types_map
                )
                
                # Define distance ranges
                distance_ranges = [
                    ('all', 0, 100),
                    ('short', 0, 5),
                    ('medium', 5, 10),
                    ('long', 10, 20)
                ]
                
                # Analyze induction energy by distance ranges
                ind_metrics = analyze_distance_ranges(
                    results_df, 'Uind_sapt', 'Uind_md', distance_ranges
                )
                
                # Analyze electrostatic energy by distance ranges
                es_metrics = analyze_distance_ranges(
                    results_df, 'Ues_sapt', 'Ues', distance_ranges
                )
                
                # Create plots if requested
                if create_plots:
                    plots_dir = os.path.join(output_dir, 'plots')
                    os.makedirs(plots_dir, exist_ok=True)
                    model_name = f"{static_model}/{drude_model}"
                    
                    # Create plots for each distance range
                    for range_name, min_dist, max_dist in distance_ranges:
                        # Filter data for this range
                        df_range = results_df[(results_df['distance'] >= min_dist) & 
                                             (results_df['distance'] < max_dist)]
                        
                        if len(df_range) > 0:
                            # Format distance range label
                            if range_name == 'all':
                                dist_label = 'all distances'
                            else:
                                dist_label = f"{min_dist}-{max_dist} Å"
                            
                            # Induction plot
                            ind_plot_path = os.path.join(
                                plots_dir,
                                f"{molecule}_{static_model}_{drude_model}_ind_{range_name}.png"
                            )
                            create_parity_plot(
                                df_range, 'Uind_sapt', 'Uind_md', ind_plot_path,
                                energy_type="induction", 
                                distance_range=dist_label,
                                molecule=molecule, model_name=model_name
                            )
                            
                            # Electrostatics plot
                            es_plot_path = os.path.join(
                                plots_dir,
                                f"{molecule}_{static_model}_{drude_model}_es_{range_name}.png"
                            )
                            create_parity_plot(
                                df_range, 'Ues_sapt', 'Ues', es_plot_path,
                                energy_type="electrostatics",
                                distance_range=dist_label,
                                molecule=molecule, model_name=model_name
                            )
                
                # Store summary results
                for range_name, metrics in ind_metrics.items():
                    all_results.append({
                        'molecule': molecule,
                        'static_model': static_model,
                        'drude_model': drude_model,
                        'energy_type': 'induction',
                        'distance_range': range_name,
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'r2': metrics['r2'],
                        'max_error': metrics['max_error'],
                        'n_points': metrics['n_points']
                    })
                
                for range_name, metrics in es_metrics.items():
                    all_results.append({
                        'molecule': molecule,
                        'static_model': static_model,
                        'drude_model': drude_model,
                        'energy_type': 'electrostatics',
                        'distance_range': range_name,
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'r2': metrics['r2'],
                        'max_error': metrics['max_error'],
                        'n_points': metrics['n_points']
                    })
                
                # Save detailed results for this combination
                detail_file = os.path.join(
                    output_dir,
                    f"{molecule}_{static_model}_{drude_model}_details.csv"
                )
                results_df.to_csv(detail_file, index=False)
                
                # Log summary
                ind_all = ind_metrics['all']
                logger.info(f"  Induction (all): MAE={ind_all['mae']:.3f}, "
                          f"RMSE={ind_all['rmse']:.3f}, R²={ind_all['r2']:.3f}")
    
    # Save summary results
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, 'parametric_study_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Study complete! Results saved to {output_dir}/")
    logger.info(f"Summary: {summary_file}")
    
    # Print best performers
    if len(summary_df) > 0:
        print_best_models(summary_df)
        
        # Generate LaTeX table
        latex_table_file = os.path.join(output_dir, 'performance_table.tex')
        generate_latex_table(summary_df, latex_table_file)
    
    return summary_df

def print_best_models(summary_df):
    """Print the best performing models based on MAE."""
    
    logger.info(f"\n{'='*60}")
    logger.info("BEST PERFORMING MODELS")
    logger.info(f"{'='*60}")
    
    # Best for induction across all distances
    ind_all = summary_df[
        (summary_df['energy_type'] == 'induction') & 
        (summary_df['distance_range'] == 'all')
    ]
    
    if len(ind_all) > 0:
        best_ind = ind_all.nsmallest(5, 'mae')
        logger.info("\nTop 5 for Induction (all distances):")
        for _, row in best_ind.iterrows():
            logger.info(f"  {row['molecule']}/{row['static_model']}/{row['drude_model']}: "
                      f"MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}")
    
    # Best for electrostatics
    es_all = summary_df[
        (summary_df['energy_type'] == 'electrostatics') & 
        (summary_df['distance_range'] == 'all')
    ]
    
    if len(es_all) > 0:
        best_es = es_all.nsmallest(5, 'mae')
        logger.info("\nTop 5 for Electrostatics (all distances):")
        for _, row in best_es.iterrows():
            logger.info(f"  {row['molecule']}/{row['static_model']}/{row['drude_model']}: "
                      f"MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}")

def generate_latex_table(summary_df, output_file=None):
    """Generate a professional LaTeX table with hierarchical structure showing MAE/RMSE for all combinations."""
    
    # Get unique values for organizing the table
    molecules = sorted(summary_df['molecule'].unique())
    static_models = sorted(summary_df['static_model'].unique())
    drude_models = sorted(summary_df['drude_model'].unique())
    distance_ranges = ['short', 'medium', 'long']  # Exclude 'all' for table clarity
    
    # Start building the LaTeX table
    latex_lines = []
    
    # Add preamble comments for required packages
    latex_lines.append("% Required packages (add to preamble if not already present):")
    latex_lines.append("% \\usepackage{booktabs, tabularx, threeparttable, makecell, multirow, siunitx}")
    latex_lines.append("% \\usepackage{caption}")
    latex_lines.append("% \\captionsetup[table]{font=small, labelfont=bf}")
    latex_lines.append("")
    latex_lines.append("% siunitx setup")
    latex_lines.append("% \\sisetup{detect-weight=true, detect-inline-weight=math, round-mode=places, round-precision=3}")
    latex_lines.append("")
    latex_lines.append("% Helper macro for metric cells")
    latex_lines.append("% \\newcommand{\\metriccell}[4]{\\makecell[l]{\\num{#1} / \\num{#2}\\\\\\num{#3} / \\num{#4}}}")
    latex_lines.append("% \\newcommand{\\rowsep}{\\addlinespace[2pt]}")
    latex_lines.append("")
    
    # Table start
    latex_lines.append("\\begin{sidewaystable*}[t]")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\small")
    latex_lines.append("  \\setlength{\\tabcolsep}{6pt}")
    latex_lines.append("  \\renewcommand{\\arraystretch}{1.2}")
    latex_lines.append("  \\begin{threeparttable}")
    latex_lines.append("    \\caption{Electrostatics and induction errors (MAE/RMSE) across charge methods, Drude models, and closest contact distance intervals.}")
    latex_lines.append("    \\label{tab:charge-drude-intervals}")
    
    # Create column specification with X columns for molecules
    col_spec = f"@{{}} l l l *{{{len(molecules)}}}{{>{{\\raggedright\\arraybackslash}}X}} @{{}}"
    latex_lines.append(f"    \\begin{{tabularx}}{{\\linewidth}}{{{col_spec}}}")
    latex_lines.append("      \\toprule")
    
    # Header rows
    latex_lines.append("      \\multicolumn{3}{c}{\\textbf{Methodology}} & " + 
                      f"\\multicolumn{{{len(molecules)}}}{{c}}{{\\textbf{{Molecule}}}} \\\\")
    latex_lines.append(f"      \\cmidrule(lr){{1-3}} \\cmidrule(lr){{4-{3+len(molecules)}}}")
    
    # Second header row
    header_parts = [
        "\\makecell[c]{Static\\\\Charge Method}",
        "\\makecell[c]{Drude\\\\Charge Method}",
        "\\makecell[c]{Distance Interval\\\\(\\AA)}"
    ]
    
    for molecule in molecules:
        display_name = molecule.replace('_', '\\_') if molecule == 'Acetic_acid' else molecule.capitalize()
        header_parts.append(f"\\makecell[c]{{\\textbf{{{display_name}}}}}")
    
    header = " &\n        ".join(header_parts) + " \\\\"
    latex_lines.append("      " + header)
    latex_lines.append("      \\midrule")
    
    # Process each static model
    for static_idx, static_model in enumerate(static_models):
        static_data = summary_df[summary_df['static_model'] == static_model]
        available_drude_models = [d for d in drude_models if len(static_data[static_data['drude_model'] == d]) > 0]
        
        if len(available_drude_models) == 0:
            continue
        
        # Calculate total rows for this static model
        total_rows = len(available_drude_models) * len(distance_ranges)
        
        # Process each drude model for this static model
        for drude_idx, drude_model in enumerate(available_drude_models):
            drude_data = static_data[static_data['drude_model'] == drude_model]
            
            # Process each distance range
            for dist_idx, distance_range in enumerate(distance_ranges):
                dist_data = drude_data[drude_data['distance_range'] == distance_range]
                
                # Create row
                row_parts = []
                
                # First column: Static model (only on first row)
                if drude_idx == 0 and dist_idx == 0:
                    if total_rows > 1:
                        row_parts.append(f"\\multirow{{{total_rows}}}{{*}}{{{static_model}}}")
                    else:
                        row_parts.append(static_model)
                else:
                    row_parts.append("")
                
                # Second column: Drude model (only on first distance range)
                if dist_idx == 0:
                    if len(distance_ranges) > 1:
                        row_parts.append(f"\\multirow{{{len(distance_ranges)}}}{{*}}{{{drude_model}}}")
                    else:
                        row_parts.append(drude_model)
                else:
                    row_parts.append("")
                
                # Third column: Distance range
                distance_labels = {'short': '0--5', 'medium': '5--10', 'long': '10--20'}
                row_parts.append(distance_labels[distance_range])
                
                # Data columns for each molecule
                for molecule in molecules:
                    mol_data = dist_data[dist_data['molecule'] == molecule]
                    
                    if len(mol_data) == 0:
                        row_parts.append("\\metriccell{--}{--}{--}{--}")
                        continue
                    
                    # Get induction and electrostatics data
                    ind_data = mol_data[mol_data['energy_type'] == 'induction']
                    es_data = mol_data[mol_data['energy_type'] == 'electrostatics']
                    
                    # Extract values
                    es_mae = es_data.iloc[0]['mae'] if len(es_data) > 0 else 0
                    es_rmse = es_data.iloc[0]['rmse'] if len(es_data) > 0 else 0
                    ind_mae = ind_data.iloc[0]['mae'] if len(ind_data) > 0 else 0
                    ind_rmse = ind_data.iloc[0]['rmse'] if len(ind_data) > 0 else 0
                    
                    # Use metriccell macro
                    row_parts.append(f"\\metriccell{{{es_mae:.3f}}}{{{es_rmse:.3f}}}{{{ind_mae:.3f}}}{{{ind_rmse:.3f}}}")
                
                # Join row parts and add to table
                row = "        " + " &\n            ".join(row_parts) + " \\\\"
                latex_lines.append(row)
                
                # Add row separator after each drude model (except the last one)
                if dist_idx == len(distance_ranges) - 1 and drude_idx < len(available_drude_models) - 1:
                    latex_lines.append("        \\rowsep")
        
        # Add midrule between static models (except after the last one)
        if static_idx < len(static_models) - 1:
            latex_lines.append("      \\midrule")
    
    # Table end
    latex_lines.append("      \\bottomrule")
    latex_lines.append("    \\end{tabularx}")
    latex_lines.append("")
    latex_lines.append("    \\begin{tablenotes}[flushleft]\\footnotesize")
    latex_lines.append("      \\item \\textit{Notes:} Values are \\textbf{MAE/RMSE} in \\si{\\kilo\\joule\\per\\mole}. Each molecule cell reports two lines:")
    latex_lines.append("      electrostatics error (top) and induction error (bottom). Distance intervals are closest contact separations in \\si{\\angstrom}.")
    latex_lines.append("      Static charge methods: " + ", ".join(static_models) + ". Drude charge methods: " + ", ".join(drude_models) + ".")
    latex_lines.append("    \\end{tablenotes}")
    latex_lines.append("  \\end{threeparttable}")
    latex_lines.append("\\end{sidewaystable*}")
    
    latex_table = "\n".join(latex_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX table saved to {output_file}")
    
    return latex_table

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run parametric force field study"
    )
    parser.add_argument(
        '--molecules', nargs='+',
        help='Molecules to test (default: all available)'
    )
    parser.add_argument(
        '--static-models', nargs='+',
        default=['ESP', 'RESP', 'MPFIT'],
        help='Static charge models to test'
    )
    parser.add_argument(
        '--drude-models', nargs='+',
        default=['2013JPC'],
        help='Drude charge models to test'
    )
    parser.add_argument(
        '--output-dir', default='parametric_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Disable plot generation'
    )
    
    args = parser.parse_args()
    
    # Run the study
    summary_df = run_parametric_study(
        molecules=args.molecules,
        static_models=args.static_models,
        drude_models=args.drude_models,
        output_dir=args.output_dir,
        create_plots=not args.no_plots
    )
    
    return 0 if len(summary_df) > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
