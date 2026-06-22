"""
Generate training/validation TOML configs for heatedCavity with randomised
boundary conditions (dirichlet or neumann, random values).
"""

import argparse
import math
import os
import random
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ─────────────────────────────────────────────────────────────────────────────
# TOML templates
# ─────────────────────────────────────────────────────────────────────────────

# Internal block boundaries (heated-cavity) are fixed at interface conditions;
# only the four outer wall BCs are randomised.
HEATED_CAVITY_TEMPLATE = """\
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

# Lid-driven cavity: square domain, all walls no-slip except north (moving lid).
# Pressure is Neumann on all walls (defined up to a constant).
LID_DRIVEN_TEMPLATE = """\
[solver.equation]
    solver = 'pressureProjection'

[solver.fluid]
    nu  = {nu}
    rho = {rho}

[solver.time]
    dt        = {dt}
    timesteps = {timesteps}

[solver.schemes]
    timeIntegrationScheme = 'secondOrderBackwards'
    nonLinearScheme       = 'secondOrderUpwind'
    diffusionScheme       = 'secondOrderCentral'

[solver.linearSolver]
    solver          = {{u = "BCGS", v = "BCGS", p = "BCGS"}}
    preconditioner  = {{u = "ILU",  v = "ILU",  p = "ILU"}}
    tolerance       = {{u = 1e-10,  v = 1e-10,  p = 1e-10}}
    maxIterations   = {{u = 1000,   v = 1000,   p = 1000}}
    underRelaxation = {{u = 1.0,    v = 1.0,    p = 1.0}}

[solver.convergence]
    picardIterations      = 1
    picard_tolerance      = {{u = 1e-3, v = 1e-3, p = 1e-3}}
    convergence_tolerance = {{u = 1e-8, v = 1e-8, p = 1e-8}}

[solver.output]
    filename         = "{case_name}"
    writingFrequency = 0

[solver.ML]
    generateTrainingData = {generate_training_data}
    trainingVariables    = ['u', 'v', 'p']

[mesh.block1]
    x = {{start = {x0}, end = {x_end}, numCells = {num_cells}}}
    y = {{start = {y0}, end = {y_end}, numCells = {num_cells}}}

[boundaryCondition.block1]
    east  = {{u = {{type = "dirichlet", value = 0.0}}, v = {{type = "dirichlet", value = 0.0}}, p = {{type = "neumann", value = 0.0}}}}
    west  = {{u = {{type = "dirichlet", value = 0.0}}, v = {{type = "dirichlet", value = 0.0}}, p = {{type = "neumann", value = 0.0}}}}
    north = {{u = {{type = "dirichlet", value = {lid_vel}}}, v = {{type = "dirichlet", value = 0.0}}, p = {{type = "neumann", value = 0.0}}}}
    south = {{u = {{type = "dirichlet", value = 0.0}}, v = {{type = "dirichlet", value = 0.0}}, p = {{type = "neumann", value = 0.0}}}}
"""

# Channel flow: rectangular domain, inlet (west) velocity prescribed,
# outlet (east) pressure fixed, north/south walls no-slip.
CHANNEL_TEMPLATE = """\
[solver.equation]
    solver = 'fsvp'

[solver.fluid]
    nu  = {nu}
    rho = {rho}

[solver.time]
    dt        = {dt}
    timesteps = {timesteps}

[solver.schemes]
    timeIntegrationScheme = 'secondOrderBackwards'
    nonLinearScheme       = 'secondOrderUpwind'
    diffusionScheme       = 'secondOrderCentral'

[solver.linearSolver]
    solver          = {{u = "BCGS", v = "BCGS", p = "BCGS"}}
    preconditioner  = {{u = "ILU",  v = "ILU",  p = "ILU"}}
    tolerance       = {{u = 1e-10,  v = 1e-10,  p = 1e-10}}
    maxIterations   = {{u = 1000,   v = 1000,   p = 1000}}
    underRelaxation = {{u = 1.0,    v = 1.0,    p = 1.0}}

[solver.convergence]
    picardIterations      = 10
    picard_tolerance      = {{u = 1e-3, v = 1e-3, p = 1e-3}}
    convergence_tolerance = {{u = 1e-8, v = 1e-8, p = 1e-8}}

[solver.output]
    filename         = "{case_name}"
    writingFrequency = 0

[solver.ML]
    generateTrainingData = {generate_training_data}
    trainingVariables    = ['u', 'v', 'p']

[mesh.block1]
    x = {{start = {x0}, end = {x_end}, numCells = {num_cells_x}}}
    y = {{start = {y0}, end = {y_end}, numCells = {num_cells_y}}}

[boundaryCondition.block1]
    east  = {{u = {{type = "neumann",   value = 0.0}}, v = {{type = "neumann",   value = 0.0}}, p = {{type = "dirichlet", value = 0.0}}}}
    west  = {{u = {{type = "dirichlet", value = {u_inlet}}}, v = {{type = "dirichlet", value = 0.0}}, p = {{type = "neumann",   value = 0.0}}}}
    north = {{u = {{type = "dirichlet", value = 0.0}}, v = {{type = "dirichlet", value = 0.0}}, p = {{type = "neumann",   value = 0.0}}}}
    south = {{u = {{type = "dirichlet", value = 0.0}}, v = {{type = "dirichlet", value = 0.0}}, p = {{type = "neumann",   value = 0.0}}}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Randomisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def random_heat_bc():
    """Dirichlet T in [0, 2] or Neumann flux in [-1, 1]."""
    bc_type = random.choice(["dirichlet", "neumann"])
    value   = round(random.uniform(0.0, 2.0), 4) if bc_type == "dirichlet" \
              else round(random.uniform(-1.0, 1.0), 4)
    return bc_type, value


def random_nu(lo=0.005, hi=0.1):
    """Log-uniform kinematic viscosity to give a spread across Reynolds numbers."""
    return round(10 ** random.uniform(math.log10(lo), math.log10(hi)), 6)


def random_lid_vel(lo=0.5, hi=2.0):
    """Lid (top-wall) velocity for lid-driven cavity."""
    return round(random.uniform(lo, hi), 4)


def random_u_inlet(lo=0.2, hi=1.5):
    """Inlet (west-wall) u-velocity for channel flow."""
    return round(random.uniform(lo, hi), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Base-config readers (extract mesh/solver defaults from existing TOMLs)
# ─────────────────────────────────────────────────────────────────────────────

def _read_heated_cavity_config(path):
    """Return defaults from a heatedCavity TOML (4-block layout)."""
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


def _read_ns_config(path):
    """Return defaults from a single-block NS TOML (lid-driven or channel)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    mesh = cfg["mesh"]["block1"]
    return {
        "num_cells_x": int(mesh["x"]["numCells"]),
        "num_cells_y": int(mesh["y"]["numCells"]),
        "x0":          float(mesh["x"]["start"]),
        "x_end":       float(mesh["x"]["end"]),
        "y0":          float(mesh["y"]["start"]),
        "y_end":       float(mesh["y"]["end"]),
        "nu":          float(cfg["solver"]["fluid"]["nu"]),
        "rho":         float(cfg["solver"]["fluid"]["rho"]),
        "dt":          float(cfg["solver"]["time"]["dt"]),
        "timesteps":   int(cfg["solver"]["time"]["timesteps"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOML writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_toml(toml_dir, case_name, content):
    path = os.path.join(toml_dir, f"{case_name}.toml")
    with open(path, "w") as f:
        f.write(content)
    return path


def write_heated_cavity_toml(toml_dir, case_name,
                              north, south, east, west,
                              north_type, south_type, east_type, west_type,
                              alpha, dt, timesteps, num_cells,
                              x0, x_split, x_end, y0, y_split, y_end,
                              generate_training_data='true'):
    content = HEATED_CAVITY_TEMPLATE.format(
        case_name=case_name,
        north=north, south=south, east=east, west=west,
        north_type=north_type, south_type=south_type,
        east_type=east_type,   west_type=west_type,
        alpha=alpha, dt=dt, timesteps=timesteps, num_cells=num_cells,
        x0=x0, x_split=x_split, x_end=x_end,
        y0=y0, y_split=y_split, y_end=y_end,
        generate_training_data=generate_training_data,
    )
    return _write_toml(toml_dir, case_name, content)


def write_lid_driven_toml(toml_dir, case_name,
                           nu, rho, lid_vel, dt, timesteps,
                           num_cells, x0, x_end, y0, y_end,
                           generate_training_data='true'):
    content = LID_DRIVEN_TEMPLATE.format(
        case_name=case_name,
        nu=nu, rho=rho, lid_vel=lid_vel,
        dt=dt, timesteps=timesteps, num_cells=num_cells,
        x0=x0, x_end=x_end, y0=y0, y_end=y_end,
        generate_training_data=generate_training_data,
    )
    return _write_toml(toml_dir, case_name, content)


def write_channel_toml(toml_dir, case_name,
                        nu, rho, u_inlet, dt, timesteps,
                        num_cells_x, num_cells_y, x0, x_end, y0, y_end,
                        generate_training_data='true'):
    content = CHANNEL_TEMPLATE.format(
        case_name=case_name,
        nu=nu, rho=rho, u_inlet=u_inlet,
        dt=dt, timesteps=timesteps,
        num_cells_x=num_cells_x, num_cells_y=num_cells_y,
        x0=x0, x_end=x_end, y0=y0, y_end=y_end,
        generate_training_data=generate_training_data,
    )
    return _write_toml(toml_dir, case_name, content)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_simulation(case_name, toml_path, check_csv):
    """Run swiftcfd and verify the expected CSV output exists."""
    ret = subprocess.run([sys.executable, "swiftcfd.py", "-i", toml_path])
    ok  = ret.returncode == 0 and os.path.exists(check_csv)
    if ok:
        print(f"  OK     — {check_csv}")
    else:
        print(f"  FAILED — {case_name} (exit {ret.returncode})")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Case generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_heated_cavity(args):
    base      = _read_heated_cavity_config(args.base_config)
    alpha     = args.alpha     if args.alpha     is not None else base["alpha"]
    dt        = args.dt        if args.dt        is not None else base["dt"]
    timesteps = args.timesteps if args.timesteps is not None else base["timesteps"]
    mesh = {k: base[k] for k in
            ("num_cells", "x0", "x_split", "x_end", "y0", "y_split", "y_end")}

    print(f"\n{'='*65}")
    print(f"  heated-cavity — {args.n_train} train / {args.n_val} val")
    print(f"  alpha={alpha}  dt={dt}  timesteps={timesteps}")
    print(f"  TOML output: {args.toml_dir}")
    print(f"{'='*65}\n")

    os.makedirs(args.toml_dir, exist_ok=True)

    # Training cases
    cases = []
    for idx in range(1, args.n_train + 1):
        case_name              = f"training_{idx:02d}"
        north_type, north = random_heat_bc()
        south_type, south = random_heat_bc()
        east_type,  east  = random_heat_bc()
        west_type,  west  = random_heat_bc()

        path = write_heated_cavity_toml(
            args.toml_dir, case_name,
            north, south, east, west,
            north_type, south_type, east_type, west_type,
            alpha, dt, timesteps, **mesh,
        )
        cases.append((case_name, path))
        print(f"  [{idx:02d}] {case_name}  "
              f"N={north_type}({north})  S={south_type}({south})  "
              f"E={east_type}({east})   W={west_type}({west})")

    if args.dry_run:
        print(f"\nDry run — {len(cases)} training TOMLs written, no simulations run.")
    else:
        print(f"\nRunning {len(cases)} swiftcfd simulations ...\n")
        failed = []
        for case_name, path in cases:
            csv = os.path.join("output", case_name, "trainingData_T.csv")
            if not _run_simulation(case_name, path, csv):
                failed.append(case_name)
        print(f"\n  Completed: {len(cases) - len(failed)}/{len(cases)} successful")
        if failed:
            print(f"  Failed: {failed}")

    # Validation cases (TOMLs only)
    print(f"\nGenerating {args.n_val} validation config(s) ...")
    for idx in range(1, args.n_val + 1):
        case_name              = f"val_{idx:02d}"
        north_type, north = random_heat_bc()
        south_type, south = random_heat_bc()
        east_type,  east  = random_heat_bc()
        west_type,  west  = random_heat_bc()

        write_heated_cavity_toml(
            args.toml_dir, case_name,
            north, south, east, west,
            north_type, south_type, east_type, west_type,
            alpha, dt, timesteps, **mesh,
            generate_training_data='false',
        )
        print(f"  [val {idx:02d}] {case_name}  "
              f"N={north_type}({north})  S={south_type}({south})  "
              f"E={east_type}({east})   W={west_type}({west})")


def generate_lid_driven(args):
    base      = _read_ns_config(args.base_config)
    dt        = args.dt        if args.dt        is not None else base["dt"]
    timesteps = args.timesteps if args.timesteps is not None else base["timesteps"]
    rho       = base["rho"]
    # For lid-driven, the mesh is always square; use num_cells for both x and y
    num_cells = base["num_cells_x"]  # assumes square
    mesh = {k: base[k] for k in ("x0", "x_end", "y0", "y_end")}

    print(f"\n{'='*65}")
    print(f"  lid-driven — {args.n_train} train / {args.n_val} val")
    print(f"  rho={rho}  dt={dt}  timesteps={timesteps}")
    print(f"  nu ∈ [{args.nu_lo}, {args.nu_hi}]  lid_vel ∈ [{args.lid_vel_lo}, {args.lid_vel_hi}]")
    print(f"  TOML output: {args.toml_dir}")
    print(f"{'='*65}\n")

    os.makedirs(args.toml_dir, exist_ok=True)

    cases = []
    for idx in range(1, args.n_train + 1):
        case_name = f"training_{idx:02d}"
        nu        = random_nu(args.nu_lo,      args.nu_hi)
        lid_vel   = random_lid_vel(args.lid_vel_lo, args.lid_vel_hi)
        re        = round(lid_vel * (mesh["x_end"] - mesh["x0"]) / nu, 1)

        path = write_lid_driven_toml(
            args.toml_dir, case_name,
            nu=nu, rho=rho, lid_vel=lid_vel,
            dt=dt, timesteps=timesteps, num_cells=num_cells, **mesh,
        )
        cases.append((case_name, path))
        print(f"  [{idx:02d}] {case_name}  nu={nu}  lid_vel={lid_vel}  Re≈{re}")

    if args.dry_run:
        print(f"\nDry run — {len(cases)} training TOMLs written, no simulations run.")
    else:
        print(f"\nRunning {len(cases)} swiftcfd simulations ...\n")
        failed = []
        for case_name, path in cases:
            csv = os.path.join("output", case_name, "trainingData_p.csv")
            if not _run_simulation(case_name, path, csv):
                failed.append(case_name)
        print(f"\n  Completed: {len(cases) - len(failed)}/{len(cases)} successful")
        if failed:
            print(f"  Failed: {failed}")

    print(f"\nGenerating {args.n_val} validation config(s) ...")
    for idx in range(1, args.n_val + 1):
        case_name = f"val_{idx:02d}"
        nu        = random_nu(args.nu_lo,      args.nu_hi)
        lid_vel   = random_lid_vel(args.lid_vel_lo, args.lid_vel_hi)
        re        = round(lid_vel * (mesh["x_end"] - mesh["x0"]) / nu, 1)

        write_lid_driven_toml(
            args.toml_dir, case_name,
            nu=nu, rho=rho, lid_vel=lid_vel,
            dt=dt, timesteps=timesteps, num_cells=num_cells, **mesh,
            generate_training_data='false',
        )
        print(f"  [val {idx:02d}] {case_name}  nu={nu}  lid_vel={lid_vel}  Re≈{re}")


def generate_channel(args):
    base      = _read_ns_config(args.base_config)
    dt        = args.dt        if args.dt        is not None else base["dt"]
    timesteps = args.timesteps if args.timesteps is not None else base["timesteps"]
    rho       = base["rho"]
    mesh = {k: base[k] for k in
            ("num_cells_x", "num_cells_y", "x0", "x_end", "y0", "y_end")}
    channel_height = mesh["y_end"] - mesh["y0"]

    print(f"\n{'='*65}")
    print(f"  channel — {args.n_train} train / {args.n_val} val")
    print(f"  rho={rho}  dt={dt}  timesteps={timesteps}")
    print(f"  nu ∈ [{args.nu_lo}, {args.nu_hi}]  u_inlet ∈ [{args.u_inlet_lo}, {args.u_inlet_hi}]")
    print(f"  TOML output: {args.toml_dir}")
    print(f"{'='*65}\n")

    os.makedirs(args.toml_dir, exist_ok=True)

    cases = []
    for idx in range(1, args.n_train + 1):
        case_name = f"training_{idx:02d}"
        nu        = random_nu(args.nu_lo,         args.nu_hi)
        u_inlet   = random_u_inlet(args.u_inlet_lo, args.u_inlet_hi)
        re        = round(u_inlet * channel_height / nu, 1)

        path = write_channel_toml(
            args.toml_dir, case_name,
            nu=nu, rho=rho, u_inlet=u_inlet,
            dt=dt, timesteps=timesteps, **mesh,
        )
        cases.append((case_name, path))
        print(f"  [{idx:02d}] {case_name}  nu={nu}  u_inlet={u_inlet}  Re≈{re}")

    if args.dry_run:
        print(f"\nDry run — {len(cases)} training TOMLs written, no simulations run.")
    else:
        print(f"\nRunning {len(cases)} swiftcfd simulations ...\n")
        failed = []
        for case_name, path in cases:
            csv = os.path.join("output", case_name, "trainingData_p.csv")
            if not _run_simulation(case_name, path, csv):
                failed.append(case_name)
        print(f"\n  Completed: {len(cases) - len(failed)}/{len(cases)} successful")
        if failed:
            print(f"  Failed: {failed}")

    print(f"\nGenerating {args.n_val} validation config(s) ...")
    for idx in range(1, args.n_val + 1):
        case_name = f"val_{idx:02d}"
        nu        = random_nu(args.nu_lo,         args.nu_hi)
        u_inlet   = random_u_inlet(args.u_inlet_lo, args.u_inlet_hi)
        re        = round(u_inlet * channel_height / nu, 1)

        write_channel_toml(
            args.toml_dir, case_name,
            nu=nu, rho=rho, u_inlet=u_inlet,
            dt=dt, timesteps=timesteps, **mesh,
            generate_training_data='false',
        )
        print(f"  [val {idx:02d}] {case_name}  nu={nu}  u_inlet={u_inlet}  Re≈{re}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Universal arguments ────────────────────────────────────────────────
    p.add_argument("--case",      choices=["heated-cavity", "lid-driven", "channel"],
                   default="heated-cavity",
                   help="Which CFD case to generate configs for (default: heated-cavity)")
    p.add_argument("--toml-dir",  default=os.path.join("input", "generated"),
                   help="Directory where generated TOML files are written")
    p.add_argument("--n-train",   type=int, default=8,
                   help="Number of training cases (default: 8)")
    p.add_argument("--n-val",     type=int, default=2,
                   help="Number of validation cases (default: 2)")
    p.add_argument("--seed",      type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--dry-run",   action="store_true",
                   help="Write TOMLs only; skip running swiftcfd")
    p.add_argument("--dt",        type=float, default=None,
                   help="Override time step (default: from base config)")
    p.add_argument("--timesteps", type=int,   default=None,
                   help="Override number of time steps (default: from base config)")

    # ── heated-cavity specific ─────────────────────────────────────────────
    p.add_argument("--base-config", default=None,
                   help="Base TOML to read mesh/solver defaults from. "
                        "Defaults: heated-cavity→input/heatedCavity.toml, "
                        "lid-driven→input/lid.toml, channel→input/channel.toml")
    p.add_argument("--alpha", type=float, default=None,
                   help="Thermal diffusivity (heated-cavity only; default: from base config)")

    # ── NS (lid-driven and channel) specific ───────────────────────────────
    p.add_argument("--nu-lo",       type=float, default=0.005,
                   help="Lower bound for kinematic viscosity (NS cases; default: 0.005)")
    p.add_argument("--nu-hi",       type=float, default=0.1,
                   help="Upper bound for kinematic viscosity (NS cases; default: 0.1)")
    p.add_argument("--lid-vel-lo",  type=float, default=0.5,
                   help="Lower bound for lid velocity (lid-driven only; default: 0.5)")
    p.add_argument("--lid-vel-hi",  type=float, default=2.0,
                   help="Upper bound for lid velocity (lid-driven only; default: 2.0)")
    p.add_argument("--u-inlet-lo",  type=float, default=0.2,
                   help="Lower bound for inlet velocity (channel only; default: 0.2)")
    p.add_argument("--u-inlet-hi",  type=float, default=1.5,
                   help="Upper bound for inlet velocity (channel only; default: 1.5)")

    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Set default base config for each case type
    if args.base_config is None:
        defaults = {
            "heated-cavity": os.path.join("input", "heatedCavity.toml"),
            "lid-driven":    os.path.join("input", "lid.toml"),
            "channel":       os.path.join("input", "channel.toml"),
        }
        args.base_config = defaults[args.case]

    if args.case == "heated-cavity":
        generate_heated_cavity(args)
    elif args.case == "lid-driven":
        generate_lid_driven(args)
    elif args.case == "channel":
        generate_channel(args)


if __name__ == "__main__":
    main()
