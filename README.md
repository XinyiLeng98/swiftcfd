# swiftcfd

![Static Badge](https://img.shields.io/badge/Version-0.27.2-blue)

> [!WARNING]  
> This project is work in progress and in version 0.X, expect breaking changes with new features.

Swiftcfd is a 2D solver for cartesian, block-structured grids for fully implicit, incompressible Navier-Stokes simulations. It has been designed around an ```Equation``` class that allows to quickly prototype new equations and test these with common Krylov subspace methods to solve the linear system $\mathbf{Ax}=\mathbf{b}$. 

## Installation

This CFD solver requires ```PETSc``` to run, which can only be compiled on UNIX. Either a UNIX environment (Linux / macOS) is required, or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows. Since ```PETSc``` is compiled from source, you will need a basic development environment installed on your PC (Python3, a C++/C compiler, and autotools).

### UNIX (WSL) installation

Using WSL or a Ubuntu-based distribution, the following commands will install the required development tools:

```bash
sudo update
sudo apt install -y build-essential
```

Once the development environment is set up, you should be able to install all Python packages with the following commands:

```bash
export PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The installation will download and compile PETSc, which can take a good 10 minutes or so, without any output being shown to the screen. If the installation fails, an error message will be printed.

When closing the terminal and opening it again, you will need to load the virtual environment again using:

```bash
source .venv/bin/activate
```

### Docker installation

A ```Dockerfile``` is provided, from which an image can be build to run the python code inside a container. To use docker, you will first need to create an image:

```
docker build -t swiftcfd .
```

We can then build the container and map the folder with the container as a volume, so code changes and outputs are synched:

```
docker run --name swiftcfd-channel -v "$(pwd):/swiftcfd" swiftcfd swiftcfd.py --input "input/channel.toml"
```

To restart to container later, use the ```docker start``` command

```
docker start -a swiftcfd-channel
```

Create as many containers as you need for different input files

## Usage

To run the solver, an input file is required. Sample input files are shown in the ```input/``` folder, which all use the [TOML](https://toml.io/en/) file extension and formatting. These input files have different section pertaining to the solver settings, the meshing, and the boundary conditions.

To execute the solver with a specific input file, use the following command:

```bash
python3 swiftcfd.py --input input/INPUT_FILE.toml
```

Here, replace ```INPUT_FILE``` by an appropriate filename, e.g. ```heatedCavitySingle.toml```, where the ```Single``` indicates that a single block is used.

## Expected results

Once a simulation has finished, results will be accumulated within the ```output``` folder. You will get, at a minimum, 3 files. The solution file as a tecplot file format, which can be read by paraview and tecplot. A residual file in ```*.csv``` format, which can be plotted in paraview or any other suitable tool, and a performance statistics file, which records the total simulation time, the average number of iterations to solve $\mathbf{Ax}=\mathbf{b}$, and the total number of timesteps.

If the ```writingFrequency``` variable within the input file is set to something greater than 0, additional files will be written during the simulation that can be loaded in paraview to create an animation of the results.



