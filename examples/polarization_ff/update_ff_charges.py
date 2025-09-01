#!/usr/bin/env python3
"""
Simple script to update XML force field files with different charge models.
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# Load charge database
with open("charge_database/charge_models.json", 'r') as f:
    CHARGE_DB = json.load(f)

def update_xml(molecule, charge_model="MPFIT", drude_model="2013JPC"):
    """Update XML file with specified charge model."""
    
    # Load molecule data
    mol_data = CHARGE_DB['molecules'][molecule]
    static_charges = mol_data['static_charges'][charge_model]
    drude_charges = mol_data['drude_charges'][drude_model]
    
    # Parse XML
    xml_path = f"{molecule}/{molecule}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get prefix (e.g., "ACA-" for acetic acid)
    first_atom = root.find('.//AtomTypes/Type')
    prefix = first_atom.get('name').split('-')[0] + '-' if '-' in first_atom.get('name') else ''
    
    # Update NonbondedForce
    nonbonded = root.find('.//NonbondedForce')
    for atom in nonbonded.findall('Atom'):
        atom_type = atom.get('type').replace(prefix, '')
        
        if atom_type in static_charges:
            # Regular atom: q_atom = q_static - q_drude
            drude_type = 'D' + atom_type
            q_static = static_charges[atom_type]
            q_drude = drude_charges.get(drude_type, 0)
            atom.set('charge', f"{q_static - q_drude:.6f}")
            
        elif atom_type.startswith('D') and atom_type in drude_charges:
            # Drude particle
            atom.set('charge', f"{drude_charges[atom_type]:.4f}")
    
    # Update DrudeForce polarizabilities
    drude_force = root.find('.//DrudeForce')
    k_drude = 0.1  # au
    au_to_nm3 = 0.000148185
    
    for particle in drude_force.findall('Particle'):
        drude_type = particle.get('type1').replace(prefix, '')
        if drude_type in drude_charges:
            q_drude = drude_charges[drude_type]
            particle.set('charge', f"{q_drude:.4f}")
            
            # Calculate polarizability: α = q_D²/k
            alpha = (q_drude**2 / k_drude) * au_to_nm3
            particle.set('polarizability', f"{alpha:.8f}")
    
    # Save to new file with proper formatting
    output_path = f"{molecule}/{molecule}_{charge_model}_{drude_model}.xml"
    
    # Pretty print the XML
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    # Remove extra blank lines
    lines = pretty_xml.split('\n')
    lines = [line for line in lines if line.strip()]
    pretty_xml = '\n'.join(lines)
    
    with open(output_path, 'w') as f:
        f.write(pretty_xml)
    
    return output_path

# Simple function for U_ind_benchmark.py integration
def get_ff_file(molecule, charge_model="MPFIT", drude_model="2013JPC"):
    """Get force field file with specified charges. Creates if needed."""
    
    # Use original file for default MPFIT/2013JPC
    # if charge_model == "MPFIT" and drude_model == "2013JPC":
    #     return f"{molecule}/{molecule}.xml"
    
    # Check if custom file exists
    custom_file = f"{molecule}/{molecule}_{charge_model}_{drude_model}.xml"
    if not os.path.exists(custom_file):
        update_xml(molecule, charge_model, drude_model)
    
    return custom_file
