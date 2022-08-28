# enerf-thesis
Repository for the codebase corresponding to my MSc thesis for UCL: Neural Radiance fields with Event Cameras

## Setup
```
conda create --name enerf python=3.8
conda activate enerf
conda install torch cudatoolkit=11.6 -c pytorch -c conda-forge
conda install numpy matplotlib tqdm scipy
```
(note: change the CUDA version may be possible)