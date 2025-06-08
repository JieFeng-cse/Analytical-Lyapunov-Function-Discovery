# Create a virtual environment with Python 3.9 (here using conda):
conda create --name alfd python=3.9
conda activate alfd

# Set up key packaging-related tools:
pip install --upgrade pip
pip install "setuptools<58.0.0"  # Required for installing deap==1.3.0

# Install dependencies:
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia ## pytroch version 2.0.1 or higher
pip install -r requirements.txt


pip install -e ./libs/sd3/dso # Install DSO package and core dependencies
cd ./libs/sd3/dso
python setup.py build_ext --inplace
cd ../../..


