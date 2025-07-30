#!/usr/bin/env python

"""
Plot SAPT vs Fortran FIT energy components from CSV files, coloring each point by distance.
Supports induction (ind20), electrostatics, and dhf comparisons.

Example usage:
  python plot_ind20_comparison.py                    # ind20 comparison (default)
  python plot_ind20_comparison.py --elec             # electrostatics comparison
  python plot_ind20_comparison.py --dhf              # dhf comparison
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os


def inset_image(ax, insetPath):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from PIL import Image
    image1 = Image.open(insetPath)

    ax_inset1 = inset_axes(ax, width="150%", height="150%", loc="upper right", bbox_to_anchor=(0.4, 0.8, 0.2, 0.2), bbox_transform=ax.transAxes)
    ax_inset1.imshow(image1)
    ax_inset1.axis('off')


def plot_fit_parity(
    csv_file,
    output_png="fit_sapt_parity.png",
    title="SAPT vs JAX/MD Parity Plot",
    chemical_accuracy_kj=1.0,
    fontsize=18,
    distance_cutoff=None,
    image_path=None,
    elec=False,
    dhf=False,
):
    """
    Reads a CSV with energy columns and plots a parity scatter.
    By default plots ind20 comparison, use flags for other components.
    Colors each point by distance, and shows a small histogram
    of errors (FIT - SAPT) in the inset.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file with columns [distance, nmer_name, Uind20, ind20_fit].
    output_png : str
        Output image filename, defaults to 'ind20_sapt_vs_fit_parity.png'.
    title : str
        Main title of the plot.
    chemical_accuracy_kj : float
        For demonstration, we draw a ± band around y=x. Adjust or remove as needed.
    fontsize : int
        Base font size for labels and text.
    """
    # 1) Read the CSV
    df = pd.read_csv(csv_file)
    
    # Determine which columns to use based on flags
    if elec:
        sapt_col = "Ues_sapt"
        fit_col = "elec_fit"
        component_name = "Electrostatics"
    elif dhf:
        sapt_col = "Udhf_sapt"
        fit_col = "dhf_fit" 
        component_name = "DHF"
    else:
        # Default is ind20
        sapt_col = "Uind20"
        fit_col = "ind20_fit"
        component_name = "ind20"
    
    # Check if required columns exist
    if fit_col not in df.columns:
        raise ValueError(f"Column '{fit_col}' not found in {csv_file}")
    if sapt_col not in df.columns:
        raise ValueError(f"Column '{sapt_col}' not found in {csv_file}")
    
    time_per_system = df['time_per_system'].iloc[0] if 'time_per_system' in df.columns else 0.0

    if distance_cutoff is not None:
        df = df[df["distance"] > distance_cutoff]
    
    # Extract columns
    x_ref = df[sapt_col].values  # SAPT values (kJ/mol)
    y_fit = df[fit_col].values  # Fortran FIT values (kJ/mol)
    color_distance = df["distance"].values

    # 2) Compute errors
    errors = y_fit - x_ref
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    r2 = np.corrcoef(x_ref, y_fit)[0, 1] ** 2
    max_error = np.max(np.abs(errors))

    # 3) Setup plot style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "xtick.labelsize": fontsize * 0.8,
            "ytick.labelsize": fontsize * 0.8,
        }
    )

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    border_width = 1.5
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(border_width)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
        direction="in",
        length=6,
        width=border_width,
        which="major",
        top=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        length=3,
        width=border_width,
        which="minor",
        top=True,
        right=True,
    )

    # 4) Axis labels & title
    ax.set_xlabel(f"SAPT {component_name} (kJ/mol)", fontsize=fontsize)
    ax.set_ylabel(f"MD/JAX Decomposed {component_name} (kJ/mol)", fontsize=fontsize)

    # 5) Plot scatter, color by distance
    sc = ax.scatter(
        x_ref, y_fit, c=color_distance, cmap="viridis", alpha=0.8, s=40, edgecolors="none"
    )
    # colorbar labeled "Distance (A)"
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Distance (Å)", fontsize=fontsize * 0.9)

    # 6) Diagonal y = x
    min_val = min(x_ref.min(), y_fit.min())
    max_val = max(x_ref.max(), y_fit.max())
    line_vals = np.linspace(min_val, max_val, 200)
    ax.plot(line_vals, line_vals, "k--", lw=2, alpha=0.7)

    # Optionally draw ± band around y=x
    lower_band = line_vals - chemical_accuracy_kj
    upper_band = line_vals + chemical_accuracy_kj
    ax.fill_between(
        line_vals,
        lower_band,
        upper_band,
        color="orange",
        alpha=0.2,
        label=f"±{chemical_accuracy_kj} kJ/mol",
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # 7) Show stats text
    text_x = 0.05
    text_y = 0.95
    stat_text = (
        f"MAE: {mae:.5f} kJ/mol\n"
        f"RMSE: {rmse:.5f} kJ/mol\n"
        f"R²: {r2:.5f}\n"
        f"Max error: {max_error:.5f} kJ/mol\n"
        f"Speed: {time_per_system:0.4f} sec/dimer"
    )
    ax.text(
        text_x,
        text_y,
        stat_text,
        transform=ax.transAxes,
        fontsize=fontsize * 0.8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # 8) Inset: error histogram
    left, bottom, width, height = 0.45, 0.15, 0.3, 0.2
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.hist(errors, bins=40, color="steelblue", alpha=0.7, edgecolor="k")
    ax_inset.set_title("Error Distribution (FIT - SAPT)", fontsize=fontsize * 0.7)
    ax_inset.tick_params(axis="both", which="major", labelsize=fontsize * 0.6)
    # Center range around zero
    half_range = max(abs(errors.min()), abs(errors.max()), chemical_accuracy_kj * 3)
    ax_inset.set_xlim(-half_range, half_range)
    ax_inset.set_yticks([])
    ax_inset.axvline(0.0, color="k", linestyle="--", linewidth=1.5)

    # Add image if provided
    if image_path is not None:
        print("ADDING IMAGE TO PLOT")
        # Determine image path relative to the CSV file if not absolute
        if not os.path.isabs(image_path):
            csv_dir = os.path.dirname(os.path.abspath(csv_file))
            full_image_path = os.path.join(csv_dir, image_path)
        else:
            full_image_path = image_path
            
        inset_image(ax, full_image_path)

    # 9) Save
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Saved {component_name} parity plot to {output_png}")


def plot_all_molecules(molecules=None, distance_cutoff=None, chemical_accuracy_kj=1.0, elec=False, dhf=False):
    """
    Plot energy component comparison for all specified molecules.
    """
    if molecules is None:
        molecules = ["imidazole", "pyrazole", "pyrazine", "acetic_acid"]
    
    # Determine component
    if elec:
        component = "elec"
        fit_col = "elec_fit"
        title_component = "Electrostatics"
    elif dhf:
        component = "dhf"
        fit_col = "dhf_fit"
        title_component = "DHF"
    else:
        component = "ind20"
        fit_col = "ind20_fit"
        title_component = "ind20"
    
    for molecule in molecules:
        csv_file = os.path.join(molecule, f"{molecule}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found for {molecule}")
            continue
        
        # Check if fit column exists
        df = pd.read_csv(csv_file)
        if fit_col not in df.columns:
            print(f"Warning: {fit_col} column not found in {molecule} data")
            continue
        
        output_png = f"results/{molecule}_{component}_comparison.png"
        image_path = f"{molecule}_mol.png"
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        plot_fit_parity(
            csv_file=csv_file,
            output_png=output_png,
            title=f"{molecule.capitalize()} SAPT vs JAX/MD {title_component}",
            chemical_accuracy_kj=chemical_accuracy_kj,
            distance_cutoff=distance_cutoff,
            image_path=image_path if os.path.exists(os.path.join(molecule, image_path)) else None,
            elec=elec,
            dhf=dhf
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAPT vs JAX/MD energy components (colored by distance)."
    )
    parser.add_argument(
        "-i", "--input_csv", 
        help="Path to a specific CSV file (if not provided, will process all molecules)."
    )
    parser.add_argument(
        "-o", "--output_png", default=None, 
        help="Output PNG filename (auto-generated based on component if not provided)."
    )
    parser.add_argument(
        "--chemical_accuracy_kj", type=float, default=1.0,
        help="Chemical accuracy in kJ/mol for the ± band (default: 1.0)."
    )
    parser.add_argument(
        "--distance_cutoff", type=float, default=None,
        help="Minimum distance cutoff in Angstroms (default: None)."
    )
    parser.add_argument(
        "--fontsize", type=int, default=18,
        help="Base font size for the plot (default: 18)."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to molecular image for inset (default: None)."
    )
    parser.add_argument(
        "--molecules", type=str, default=None,
        help="Comma-separated list of molecules to process (default: all)."
    )
    parser.add_argument(
        "--elec", action="store_true",
        help="Plot electrostatics comparison instead of ind20."
    )
    parser.add_argument(
        "--dhf", action="store_true",
        help="Plot DHF comparison instead of ind20."
    )

    args = parser.parse_args()

    if args.input_csv:
        # Process single CSV file
        # Auto-generate output filename if not provided
        if args.output_png is None:
            if args.elec:
                args.output_png = "elec_sapt_vs_fit_parity.png"
            elif args.dhf:
                args.output_png = "dhf_sapt_vs_fit_parity.png"
            else:
                args.output_png = "ind20_sapt_vs_fit_parity.png"
        
        plot_fit_parity(
            csv_file=args.input_csv,
            output_png=args.output_png,
            chemical_accuracy_kj=args.chemical_accuracy_kj,
            fontsize=args.fontsize,
            distance_cutoff=args.distance_cutoff,
            image_path=args.image,
            elec=args.elec,
            dhf=args.dhf
        )
    else:
        # Process all molecules
        molecules = args.molecules.split(',') if args.molecules else None
        plot_all_molecules(
            molecules=molecules,
            distance_cutoff=args.distance_cutoff,
            chemical_accuracy_kj=args.chemical_accuracy_kj,
            elec=args.elec,
            dhf=args.dhf
        )


if __name__ == "__main__":
    main()
