# Creating a python virtual environment: 
python3 -m venv amc_env 

# Activate the environment:
source amc_env/bin/activate

# Install dependencies before building LPMP:
pip install wheel
pip install numpy
pip install matplotlib

# Install LPMP python bindings:
python3 -m pip install .