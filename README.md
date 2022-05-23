# Extreme Methods

## SETUP LIBRARY
```bash
mkdir -p ${HOME}/scratch/XC/programs
cd ${HOME}/scratch/XC/programs
git clone https://github.com/anshumitts/ExtremeMethods.git
conda create -f ExtremeMethods/CafeXC.yml
conda activate xc
pip install hnswlib Cython git+https://github.com/kunaldahiya/pyxclib.git
```

## ALGORITHMS IMPLEMENTED
- MUFIN - Multimodal extreme classification - [link](https://github.com/Extreme-classification/MUFIN)
