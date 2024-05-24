# Scalable Angular Spectrum Method SAS
Implementations of the Scalable Angular Spectrum (SAS) method for optical wave propagation in Python + PyTorch and Julia Lang.
The SAS allows to propagate optical waves accurately over long distances and high magnifications. 
It features a much higher speed than the Angular Spectrum (AS) method and is more precise than the single-step Fresnel (SFT-Fr) method.
It's implementation essentially consists of three FFTs and some phase factors.

This image compares the three methods for accuracy.
![](example.png)

The calculation time for Fresnel was 0.04s, the one for SAS around 0.12s. But AS takes around 6s because of large padding!

# Publication and Citation
For more details about the mathematics and physics behind this method, see this [OPTICA publication](https://opg.optica.org/optica/fulltext.cfm?uri=optica-10-11-1407&id=541154).
Please consider citing this, if you find it useful:
```
@article{Heintzmann:23,
author = {Rainer Heintzmann and Lars Loetgering and Felix Wechsler},
journal = {Optica},
keywords = {Fourier transforms; Inverse design; Phase retrieval; Spatial frequency; Synthetic aperture radar; Wave propagation},
number = {11},
pages = {1407--1416},
publisher = {Optica Publishing Group},
title = {Scalable angular spectrum propagation},
volume = {10},
month = {Nov},
year = {2023},
url = {https://opg.optica.org/optica/abstract.cfm?URI=optica-10-11-1407},
doi = {10.1364/OPTICA.497809}
}
```

# License
Check the conditions of the [license](LICENSE).


# Julia Code
Tested with [Julia 1.9](https://julialang.org/downloads/). Download the software and install it on your computer. 
Open the REPL and type (type really a `]` to enter the package manager. Delete the `]` to go back to REPL):
```julia
julia > ]add Pluto

julia > using Pluto
```
[Pluto.jl](https://github.com/fonsp/Pluto.jl) is an interactive notebook environment.
Download the file [SAS_examples.jl](SAS_examples.jl). From within Pluto, open this file.
Everything should go fine from now on.

## Reproducibility
Pluto notebooks are highly reproducible because of the pinned versions of all dependencies. So running this notebook with Julia 1.9.1 is reproducible. You do not have to handle the versions yourself. Everything is done by Pluto.jl.

# Python
Implementation in Python and [PyTorch](https://pytorch.org/). Hence, it can be used within neural networks and automatic differentiation.
See this [Jupyter notebook](SAS_pytorch.ipynb).
Install
```
pip install numpy matplotlib torch notebook
```
to run the Python notebook.

# Implementations
There is an efficient implementation of the Angular Spectrum method and the Scalable Angular Spectrum method in [WaveOpticsPropagation.jl](https://github.com/JuliaPhysics/WaveOpticsPropagation.jl/)


# Bugs, Issues
Please feel free to file issues and bugs here on GitHub! Also, if you have any question regarding the paper, post it here too!
