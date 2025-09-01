#!/usr/bin/env python3
"""
Minimal force field generator that uses template XMLs and charge database.
"""

import json
import xml.etree.ElementTree as ET
import os
import sys

def update_ff(molecule, charge_model="MPFIT", drude_model="2013JPC"):
    """Generate force field XML with specified charges."""
    
    # Load charges
    with open("charge_database/charge_models_v3.json", 'r') as f:
        db = json.load(f)
    
    if molecule not in db['molecules']:
        raise ValueError(f"Molecule {molecule} not in database")
    
    mol_data = db['molecules'][molecule]
    static = mol_data['static_charges'].get(charge_model, {})
    drude = mol_data['drude_charges'].get(drude_model, {})
    
    if not static:
        raise ValueError(f"No {charge_model} charges for {molecule}")
    
    # Load and parse template XML
    template_path = f"{molecule}/{molecule}_minimal.xml"
    tree = ET.parse(template_path)
    root = tree.getroot()
    
    # Get prefix from first atom type
    first_type = root.find('.//AtomTypes/Type')
    prefix = ''
    if first_type is not None and '-' in first_type.get('name', ''):
        prefix = first_type.get('name').split('-')[0] + '-'
    
    # Update NonbondedForce charges
    for atom in root.findall('.//NonbondedForce/Atom'):
        atom_type = atom.get('type').replace(prefix, '')
        
        if atom_type.startswith('D'):
            # Drude particle
            if atom_type in drude:
                atom.set('charge', f"{drude[atom_type]:.4f}")
        else:
            # Regular atom
            if atom_type in static:
                q_static = static[atom_type]
                drude_type = f'D{atom_type}'
                # Subtract drude charge (drude charges are negative, so subtracting makes parent more positive)
                q_eff = q_static - drude.get(drude_type, 0)
                atom.set('charge', f"{q_eff:.4f}")
    
    # Update DrudeForce
    k_drude = db['constants']['k_drude']
    au_to_nm3 = db['constants']['au_to_nm3']
    
    for particle in root.findall('.//DrudeForce/Particle'):
        drude_type = particle.get('type1').replace(prefix, '')
        if drude_type in drude:
            q_d = drude[drude_type]
            particle.set('charge', f"{q_d:.4f}")
            # Polarizability = q_d^2 / k * conversion
            alpha = (q_d**2 / k_drude) * au_to_nm3
            particle.set('polarizability', f"{alpha:.8f}")
    
    # Write output
    output_path = f"{molecule}/{molecule}_{charge_model}_{drude_model}.xml"
    tree.write(output_path, encoding='unicode', xml_declaration=True)
    
    print(f"Generated: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ff_generator.py <molecule> [charge_model] [drude_model]")
        sys.exit(1)
    
    mol = sys.argv[1]
    charge = sys.argv[2] if len(sys.argv) > 2 else "MPFIT"
    drude = sys.argv[3] if len(sys.argv) > 3 else "2013JPC"
    
    try:
        update_ff(mol, charge, drude)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
