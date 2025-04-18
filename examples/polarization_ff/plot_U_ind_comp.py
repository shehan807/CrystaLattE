#!/usr/bin/env python

"""
Plot SAPT vs MD induction energies from a CSV, coloring each point by distance.

Example usage:
  python plot_induction_parity.py -i induction_data.csv -o parity_plot.png
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


def plot_induction_parity(
    csv_file,
    output_png="sapt_vs_md_parity.png",
    title="SAPT vs MD Induction Parity Plot",
    chemical_accuracy_kj=1.0,
    fontsize=18,
    distance_cutoff=None,
    verification=False,
    omm=False,
    nonbondedforce=False,
    drudeforce=False,
    electrostatics=False,
    image_path=None,  
):
    """
    Reads a CSV with columns:
        distance, nmer_name, Uind_md, Uind_sapt
    in kJ/mol units, then plots a parity scatter:
      x = Uind_sapt (kJ/mol)
      y = Uind_md   (kJ/mol)
    Colors each point by distance, and shows a small histogram
    of errors (MD - SAPT) in the inset.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file with columns [distance, nmer_name, Uind_md, Uind_sapt].
    output_png : str
        Output image filename, defaults to 'sapt_vs_md_parity.png'.
    title : str
        Main title of the plot.
    chemical_accuracy_kj : float
        For demonstration, we draw a ± band around y=x. Adjust or remove as needed.
    fontsize : int
        Base font size for labels and text.
    """
    # 1) Read the CSV
    df = pd.read_csv(csv_file)
    time_per_system = df['time_per_system'].iloc[0]

    if distance_cutoff is not None:
        df = df[df["distance"] > distance_cutoff]
    # Extract columns
    if verification:
        x_ref = df["Uind_omm"].values  # (kJ/mol)
    elif drudeforce:
        x_ref = df["Udf_omm"].values  # (kJ/mol)
    elif nonbondedforce:
        x_ref = df["Unb_omm"].values  # (kJ/mol)
    elif electrostatics:
        x_ref = df["Ues_sapt"].values  # (kJ/mol)
    else:
        x_ref = df["Uind_sapt"].values  # (kJ/mol)
    
    if omm:
        y_md = df["Uind_omm"].values      # (kJ/mol)
    elif drudeforce:
        y_md = df["Udf"].values  # (kJ/mol)
    elif nonbondedforce:
        y_md = df["Unb"].values  # (kJ/mol)
    elif electrostatics:
        y_md = df["Ues"].values  # (kJ/mol)
    else:
        y_md = df["Uind_md"].values      # (kJ/mol)
    color_distance = df["distance"].values

    # 2) Compute errors
    errors = y_md - x_ref
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    r2 = np.corrcoef(x_ref, y_md)[0, 1] ** 2
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
    if verification:
        ax.set_xlabel("OpenMM Induction (kJ/mol)", fontsize=fontsize)
    elif drudeforce:
        ax.set_xlabel("OpenMM DrudeForce (kJ/mol)", fontsize=fontsize)
    elif nonbondedforce:
        ax.set_xlabel("OpenMM NonbondedForce (kJ/mol)", fontsize=fontsize)
    elif electrostatics:
        ax.set_xlabel("SAPT Electrostatics (kJ/mol)", fontsize=fontsize)
    else:
        ax.set_xlabel("SAPT Induction (kJ/mol)", fontsize=fontsize)
    if omm:
        ax.set_ylabel("OpenMM Induction (kJ/mol)", fontsize=fontsize)
    elif drudeforce:
        ax.set_ylabel("OpenMM DrudeForce (kJ/mol)", fontsize=fontsize)
    elif nonbondedforce:
        ax.set_ylabel("OpenMM NonbondedForce (kJ/mol)", fontsize=fontsize)
    elif electrostatics:
        ax.set_ylabel("MD Electrostatics (kJ/mol)", fontsize=fontsize)
    else:
        ax.set_ylabel("MD Induction (kJ/mol)", fontsize=fontsize)
    # ax.set_title(title, fontsize=fontsize)

    # 5) Plot scatter, color by distance
    sc = ax.scatter(
        x_ref, y_md, c=color_distance, cmap="viridis", alpha=0.8, s=40, edgecolors="none"
    )
    # print(x_ref, y_md, color_distance)
    # colorbar labeled "Distance (A)"
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Distance (Å)", fontsize=fontsize * 0.9)

    # 6) Diagonal y = x
    min_val = min(x_ref.min(), y_md.min())
    max_val = max(x_ref.max(), y_md.max())
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
    ax_inset.set_title("Error Distribution (MD - SAPT)", fontsize=fontsize * 0.7)
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
    print(f"Saved parity plot to {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAPT vs MD induction energies (colored by distance)."
    )
    parser.add_argument(
        "-i", "--input_csv", required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output_png", default="sapt_vs_md_parity.png", 
        help="Output PNG filename (default: sapt_vs_md_parity.png)."
    )
    parser.add_argument(
        "--title",
        default="SAPT vs MD Induction Parity Plot",
        help="Title for the plot (default: 'SAPT vs MD Induction Parity Plot').",
    )
    parser.add_argument(
        "--chemical_accuracy_kj",
        type=float,
        default=1.0,
        help="± band drawn around y=x (kJ/mol). Default=1.0",
    )
    parser.add_argument(
        "--fontsize", type=int, default=18, help="Base font size for labels/text."
    )
    parser.add_argument(
        "--distance_cutoff", type=float, default=None,
        help="Only plot values at distances higher than this cutoff."
    )
    parser.add_argument(
        "-v", "--verification", action="store_true", default=False,
        help="Enable verification mode (default: False)."
    ) 
    parser.add_argument(
        "-omm", "--openmm", action="store_true", default=False,
        help="Set y_md to OpenMM Uind mode (default: False)."
    ) 
    parser.add_argument(
        "-df", "--drudeforce", action="store_true", default=False,
        help="Enable DrudeForce comparison (default: False)."
    ) 
    parser.add_argument(
        "-nb", "--nonbondedforce", action="store_true", default=False,
        help="Enable NonbondedForce comparison (default: False)."
    ) 
    parser.add_argument(
        "-es", "--electrostatics", action="store_true", default=False,
        help="Enable SAPT (Electrostatics) comparison (default: False)."
    ) 
    parser.add_argument(
        "--image",
        help="Path to an image file to add to the plot. Can be relative to CSV file."
    )
    args = parser.parse_args()

    plot_induction_parity(
        csv_file=args.input_csv,
        output_png=args.output_png,
        title=args.title,
        chemical_accuracy_kj=args.chemical_accuracy_kj,
        fontsize=args.fontsize,
        distance_cutoff=args.distance_cutoff,
        verification=args.verification,
        omm=args.openmm,
        drudeforce=args.drudeforce,
        nonbondedforce=args.nonbondedforce,
        electrostatics=args.electrostatics,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()

