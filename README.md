# CellOrigin

&nbsp;
  
## Prerequisites
* [minepy ver.1.2.6](https://github.com/minepy/minepy)
* numpy < 2.0 (Currently, mienpy doesn't work with numpy 2.0 or higher version)


&nbsp;

## Installation
```
git clone https://github.com/Irrationall/CellOrigin

cd CellOrigin

pip install .

# install in editable mode:
pip install -e .
```

&nbsp;

## Tutorial

&nbsp;

## Compatibility with numpy 2.0
In **Scanpy 1.10.3**, **NumPy 2.0** is compatible by default. However, **Minepy** is not initially compatible with NumPy 2.0. To resolve this, you need to recompile **Minepy** in an environment where **NumPy 2.0** is installed.

This guide provides a step-by-step workflow to recompile **Minepy** for compatibility with **NumPy 2.0**.

---

* Step 1: Set up a virtual environment. (Here we used conda)
```bash
conda create -n newminepy python=3.10
conda activate newminepy
```
* Step 2: Install **Scanpy** and clone **Minepy** repository. The recent version of **Scanpy** automatically installs **NumPy > 2.0**
```bash
(newminepy) pip install scanpy
(newminepy) git clone https://github.com/minepy/minepy.git
```
* Step 3: Move on to **Minepy** directory and compile it.
```bash
# Make new c file
(newminepy) cd minepy 
(newminepy) pip install cython
(newminepy) bash compile_pyx.sh
```
A new mine.c file is created in the ./minepy directory.
```bash
# compile c file
(newminepy) gcc -shared -pthread -fPIC -fwrapv -O2 \
                -I/opt/anaconda/envs/newminepy/include/python3.10 \
                -I/opt/anaconda/envs/newminepy/lib/python3.10/site-packages/numpy/_core/include \
                -o minepy/mine.new.so minepy/mine.c

# Install minepy
(newminepy) pip install .
```
* Step 4: Check **Minepy** works well
```bash
# Go outside of minepy folder
(newminepy) cd ~
(newminepy) python
```
```python
from minepy import MINE
import numpy as np

print(f'NumPy version: {np.__version__}')
```
NumPy version: 2.0.2
```python
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]

mine = MINE()
mine.compute_score(x, y)
res = [mine.mic(), mine.mas(), mine.mev(), mine.mcn(), mine.tic()]
res
```
[0.9709505944546685, 0.0, 0.9709505944546685, 2.0, 0.9709505944546685]

&nbsp;

## Citation
* Paper
```
TBD
```
* [Relative Dimension](https://github.com/barahona-research-group/DynGDim)
```
@article{peach2022relative,
  title={Relative, local and global dimension in complex networks},
  author={Peach, Robert and Arnaudon, Alexis and Barahona, Mauricio},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={3088},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```
