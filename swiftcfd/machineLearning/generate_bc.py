"""
Generate training/validation TOML configs for heatedCavity with randomised
boundary conditions (dirichlet or neumann, random values).
"""

import argparse
import os
import random
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Interface BCs (internal block boundaries) are fixed; only outer wall types vary.
TOML_TEMPLATE = """\
[solver.equation]
    solver = 'heatDiffusion'

[solver.fluid]
    alpha = {alpha}

[solver.time]
    dt        = {dt}
    timesteps = {timesteps}

[solver.schemes]
    timeIntegrationScheme = 'secondOrderBackwards'
    diffusionScheme       = 'secondOrderCentral'

[solver.linearSolver]
    solver          = {{T = "GMRES"}}
    preconditioner  = {{T = "ILU"}}
    tolerance       = {{T = 1e-12}}
    maxIterations   = {{T = 1000}}
    underRelaxation = {{T = 1.0}}

[solver.convergence]
    picardIterations      = 1
    picard_tolerance      = {{T = 1}}
    convergence_tolerance = {{T = 1e-10}}

[solver.output]
    filename         = "{case_name}"
    writingFrequency = 0

[solver.ML]
    generateTrainingData = {generate_training_data}
    trainingVariables    = ['T']

[mesh.block1]
    x = {{start = {x0}, end = {x_split}, numCells = {num_cells}}}
    y = {{start = {y0}, end = {y_split}, numCells = {num_cells}}}

[mesh.block2]
    x = {{start = {x_split}, end = {x_end}, numCells = {num_cells}}}
    y = {{start = {y0}, end = {y_split}, numCells = {num_cells}}}

[mesh.block3]
    x = {{start = {x0}, end = {x_split}, numCells = {num_cells}}}
    y = {{start = {y_split}, end = {y_end}, numCells = {num_cells}}}

[mesh.block4]
    x = {{start = {x_split}, end = {x_end}, numCells = {num_cells}}}
    y = {{start = {y_split}, end = {y_end}, numCells = {num_cells}}}

[boundaryCondition.block1]
    east  = {{T = {{type = "interface",    value = 2}}}}
    west  = {{T = {{type = "{west_type}",  value = {west}}}}}
    north = {{T = {{type = "interface",    value = 3}}}}
    south = {{T = {{type = "{south_type}", value = {south}}}}}

[boundaryCondition.block2]
    east  = {{T = {{type = "{east_type}",  value = {east}}}}}
    west  = {{T = {{type = "interface",    value = 1}}}}
    north = {{T = {{type = "interface",    value = 4}}}}
    south = {{T = {{type = "{south_type}", value = {south}}}}}

[boundaryCondition.block3]
    east  = {{T = {{type = "interface",     value = 4}}}}
    west  = {{T = {{type = "{west_type}",   value = {west}}}}}
    north = {{T = {{type = "{north_type}",  value = {north}}}}}
    south = {{T = {{type = "interface",     value = 1}}}}

[boundaryCondition.block4]
    east  = {{T = {{type = "{east_type}",  value = {east}}}}}
    west  = {{T = {{type = "interface",    value = 3}}}}
    north = {{T = {{type = "{north_type}", value = {north}}}}}
    south = {{T = {{type = "interface",    value = 2}}}}
"""


def random_bc():
    """Return (type, value): dirichlet T in [0, 2] or neumann flux in [-1, 1]."""
    bc_type = random.choice(["dirichlet", "neumann"])
    if bc_type == "dirichlet":
        value = round(random.uniform(0.0, 2.0), 4)
    else:
        value = round(random.uniform(-1.0, 1.0), 4)
    return bc_type, value


def write_toml(toml_dir, case_name,
               north, south, east, west,
               north_type, south_type, east_type, west_type,
               alpha, dt, timesteps, num_cells,
               x0, x_split, x_end, y0, y_split, y_end,
               generate_training_data='true'):
    content = TOML_TEMPLATE.format(
        case_name=case_name,
        north=north, south=south, east=east, west=west,
        north_type=north_type, south_type=south_type,
        east_type=east_type,   west_type=west_type,
        alpha=alpha, dt=dt, timesteps=timesteps,
        num_cells=num_cells,
        x0=x0, x_split=x_split, x_end=x_end,
        y0=y0, y_split=y_split, y_end=y_end,
        generate_training_data=generate_training_data,
    )
    path = os.path.join(toml_dir, f"{case_name}.toml")
    with open(path, "w") as f:
        f.write(content)
    return path


def _read_base_config(path):
    """Read mesh and solver defaults from the base heatedCavity TOML."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    mesh = cfg["mesh"]
    b1, b2, b4 = mesh["block1"], mesh["block2"], mesh["block4"]
    return {
        "num_cells": int(b1["x"]["numCells"]),
        "x0":        float(b1["x"]["start"]),
        "x_split":   float(b1["x"]["end"]),
        "x_end":     float(b2["x"]["end"]),
        "y0":        float(b1["y"]["start"]),
        "y_split":   float(b1["y"]["end"]),
        "y_end":     float(b4["y"]["end"]),
        "alpha":     float(cfg["solver"]["fluid"]["alpha"]),
        "dt":        float(cfg["solver"]["time"]["dt"]),
        "timesteps": int(cfg["solver"]["time"]["timesteps"]),
    }


def main():
    p = argparse.ArgumentParser(
        description="Generate heatedCavity BC variants with random types and values"
    )
    p.add_argument("--base-config", default=os.path.join("input", "heatedCavity.toml"))
    p.add_argument("--alpha",     type=float, default=None)
    p.add_argument("--dt",        type=float, default=None)
    p.add_argument("--timesteps", type=int,   default=None)
    p.add_argument("--toml-dir",  default=os.path.join("input", "generated"))
    p.add_argument("--n-train",   type=int,   default=8,
                   help="Number of training cases (default: 8)")
    p.add_argument("--n-val",     type=int,   default=2,
                   help="Number of validation cases (default: 2)")
    p.add_argument("--seed",      type=int,   default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--dry-run",   action="store_true",
                   help="Write TOMLs only; skip running swiftcfd")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    base      = _read_base_config(args.base_config)
    alpha     = args.alpha     if args.alpha     is not None else base["alpha"]
    dt        = args.dt        if args.dt        is not None else base["dt"]
    timesteps = args.timesteps if args.timesteps is not None else base["timesteps"]
    mesh_kwargs = {k: base[k] for k in
                   ("num_cells", "x0", "x_split", "x_end", "y0", "y_split", "y_end")}

    os.makedirs(args.toml_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  heatedCavity BC generator — {args.n_train} train / {args.n_val} val")
    print(f"  alpha={alpha}  dt={dt}  timesteps={timesteps}")
    print(f"  TOML output: {args.toml_dir}")
    print(f"{'='*65}\n")

    # ── Training cases ────────────────────────────────────────────────────────
    cases = []
    for idx in range(1, args.n_train + 1):
        case_name                  = f"training_{idx:02d}"
        north_type, north = random_bc()
        south_type, south = random_bc()
        east_type,  east  = random_bc()
        west_type,  west  = random_bc()

        toml_path = write_toml(
            args.toml_dir, case_name,
            north, south, east, west,
            north_type, south_type, east_type, west_type,
            alpha, dt, timesteps, **mesh_kwargs,
        )
        cases.append((case_name, toml_path))
        print(f"  [{idx:02d}] {case_name}  "
              f"N={north_type}({north})  S={south_type}({south})  "
              f"E={east_type}({east})  W={west_type}({west})")

    if args.dry_run:
        print(f"\nDry run — {len(cases)} training TOMLs written, no simulations run.")
    else:
        print(f"\nRunning {len(cases)} swiftcfd simulations ...\n")
        failed = []
        for case_name, toml_path in cases:
            ret      = subprocess.run([sys.executable, "swiftcfd.py", "-i", toml_path])
            csv_path = os.path.join("output", case_name, "trainingData_T.csv")
            if ret.returncode != 0 or not os.path.exists(csv_path):
                print(f"  FAILED — {case_name} (exit {ret.returncode})")
                failed.append(case_name)
            else:
                print(f"  OK — {csv_path}")

        print(f"\n  Completed: {len(cases) - len(failed)}/{len(cases)} successful")
        if failed:
            print(f"  Failed: {failed}")

    # ── Validation cases (TOMLs only, no simulation) ─────────────────────────
    print(f"\nGenerating {args.n_val} validation config(s) ...")
    for idx in range(1, args.n_val + 1):
        case_name                  = f"val_{idx:02d}"
        north_type, north = random_bc()
        south_type, south = random_bc()
        east_type,  east  = random_bc()
        west_type,  west  = random_bc()

        toml_path = write_toml(
            args.toml_dir, case_name,
            north, south, east, west,
            north_type, south_type, east_type, west_type,
            alpha, dt, timesteps, **mesh_kwargs,
            generate_training_data='false',
        )
        print(f"  [val {idx:02d}] {case_name}  "
              f"N={north_type}({north})  S={south_type}({south})  "
              f"E={east_type}({east})  W={west_type}({west})")


if __name__ == "__main__":
    main()
