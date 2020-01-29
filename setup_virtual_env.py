import os
import spectra_generator.spectra_generator

"""
This is a utility to install matlab.engine in your python virtual environment
"""

# In matlab command window, enter '`matlabroot`' and copy to this variable:
MATLABROOT = "/Applications/MATLAB_R2019b.app/extern/engines/python"
VENV_DIRNAME = "venv"

project_root = os.path.dirname(os.path.realpath(__file__))

# First run matlab setup
os.chdir(MATLABROOT)
os.system('python3 setup.py install --prefix="%s"' % os.path.join(project_root, VENV_DIRNAME))


# Copy this into venv/activate
print("\n\nCopy this into your venv/activate file:")
print('export PYTHONPATH="%s"' % project_root)
