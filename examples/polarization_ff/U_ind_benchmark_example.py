"""
Example showing how to modify U_ind_benchmark.py to use different charge models.
"""

# In U_ind_benchmark.py, you would modify the main() function like this:

# Add import at the top:
from update_ff_charges import get_ff_file

# Then in the main loop, change this line:
# ff_file = os.path.join(molecule, molecule + ".xml")
ff_file = get_ff_file("acetic_acid", charge_model="MPFIT", drude_model="2013JPC")

# To this (for ESP charges with 2013JPC drude charges):
ff_file = get_ff_file("acetic_acid", charge_model="ESP", drude_model="2013JPC")

# Or for RESP charges:
ff_file = get_ff_file("acetic_acid", charge_model="RESP", drude_model="2013JPC")

# The rest of the code remains exactly the same!
