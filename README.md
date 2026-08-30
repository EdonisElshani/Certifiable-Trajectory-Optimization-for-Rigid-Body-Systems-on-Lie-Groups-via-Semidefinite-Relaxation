# Certifiable Trajectory Optimization for Rigid-Body Systems on Lie Groups via Semidefinite Relaxation

This repository contains the numerical simulations and trajectory-optimization models developed for a master's thesis on structure-preserving and certifiable trajectory optimization of rigid-body systems on Lie groups.

Two benchmark systems are considered:

- a three-dimensional pendulum evolving on $\mathrm{SO}(3)$;
- a planar Acrobot evolving on $\mathrm{SO}(2)\times\mathrm{SO}(2)$.

The numerical simulations compare Lie group variational integrators (LGVIs) with Runge--Kutta methods. The optimization models formulate the finite-horizon dynamics as polynomial optimization problems and construct sparse Moment--SOS relaxations with [SPOT](https://github.com/ComputationalRobotics/SPOT).

> [!NOTE]
> An SDP relaxation provides a lower bound. A global-optimality certificate additionally requires a verified feasible trajectory providing a matching upper bound. Consequently, not every optimization or MPC run in this repository is certified.

## Thesis

The final thesis PDF will be available here after it is added to the repository:

**[Download the master's thesis](./Edonis_Elshani_Master_Thesis.pdf)**

Upload the PDF to the repository root with the filename `Edonis_Elshani_Master_Thesis.pdf` to activate this link.

## Repository structure

```text
.
├── Numerical_Simulation/
│   ├── main_numerical_simulation.py
│   ├── solver_lgvi.py
│   ├── solver_lgvi_cayley.py
│   ├── solver_rk4.py
│   ├── Discrete_Mechanical_Models_Lie_Groups/
│   │   ├── model_3d_pendulum.py
│   │   ├── model_3d_pendulum_forced.py
│   │   └── model_acrobot_so2.py
│   └── Acrobot/
│       ├── main_acrobot_unforced.py
│       ├── main_acrobot_forced_sine.py
│       ├── solver_lgvi_acrobot.py
│       └── solver_rk4_acrobot.py
└── Ridged_Body_SDP/
    ├── pendulum_SO3_Python.py
    └── Acrobot_SO2_Python/
        ├── open_loop_sdp.py
        ├── main_sdp_mpc.py
        ├── simulation.py
        ├── config/
        ├── SDP/
        └── Numerical_Simulation/
```

`Ridged_Body_SDP` is retained here because it is the current directory name in the repository.

## Numerical simulations

The standalone simulations require Python together with NumPy, SciPy, and Matplotlib:

```bash
python -m pip install numpy scipy matplotlib pyyaml
```

### 3D pendulum

Run the main comparison from the repository root:

```bash
python Numerical_Simulation/main_numerical_simulation.py
```

The script compares the LGVI with RK4 and reports rotation-matrix orthogonality, energy deviation, the gravity-axis momentum error, and the residual of the implicit LGVI solve. The controlled and unforced models can be selected near the bottom of `main_numerical_simulation.py`:

```python
model = ForcedPendulum3DModel()
# model = Pendulum3DModel()
```

### Acrobot

Run the unforced or sinusoidally forced Acrobot simulation from the repository root:

```bash
python Numerical_Simulation/Acrobot/main_acrobot_unforced.py
python Numerical_Simulation/Acrobot/main_acrobot_forced_sine.py
```

These scripts compare the maximal-coordinate LGVI with RK4-based simulations and evaluate trajectory differences, rotation-matrix orthogonality, energy behavior, and holonomic-constraint residuals. Generated data and plots are written below `Numerical_Simulation/Acrobot/output/`.

## Polynomial optimization models

### 3D pendulum on $\mathrm{SO}(3)$

`Ridged_Body_SDP/pendulum_SO3_Python.py` defines the controlled finite-horizon pendulum problem using polynomial rotation, kinematic, dynamics, step-angle, and control constraints. It constructs a sparse Moment--SOS relaxation and extracts candidate trajectories from the moment solution.

### Acrobot on $\mathrm{SO}(2)\times\mathrm{SO}(2)$

`Ridged_Body_SDP/Acrobot_SO2_Python/` contains:

- the reduced polynomial Acrobot model;
- open-loop SDP trajectory optimization;
- the full maximal-coordinate numerical plant;
- an SDP-based receding-horizon controller;
- YAML configuration files for physical, solver, and MPC parameters;
- logging, diagnostics, and plotting utilities.

Run the open-loop or MPC entry points after completing the SPOT setup described below.

## SPOT installation and model placement

The SDP models use the [Sparse Polynomial Optimization Toolbox (SPOT)](https://github.com/ComputationalRobotics/SPOT). SPOT is not included in this repository and must be installed separately together with its solver dependencies, including MOSEK. Follow the current Python installation instructions in the SPOT repository before running the optimization models.

After installing SPOT, copy the following files into SPOT's `Python-examples/` directory:

```text
Ridged_Body_SDP/pendulum_SO3_Python.py
Ridged_Body_SDP/Acrobot_SO2_Python/
```

The resulting SPOT tree should contain:

```text
SPOT/
├── SPOT/
├── Python-examples/
│   ├── pendulum_SO3_Python.py
│   └── Acrobot_SO2_Python/
└── ...
```

From the root of the cloned SPOT repository, run:

```bash
python Python-examples/pendulum_SO3_Python.py
python Python-examples/Acrobot_SO2_Python/open_loop_sdp.py
python Python-examples/Acrobot_SO2_Python/main_sdp_mpc.py
```

To use another Acrobot configuration:

```bash
python Python-examples/Acrobot_SO2_Python/main_sdp_mpc.py \
  --config Python-examples/Acrobot_SO2_Python/config/acrobot_physical.yaml
```

Solver outputs, extracted trajectories, moment-matrix diagnostics, and MPC histories are stored in directories created by the corresponding scripts.

## Method summary

The workflow used by the optimization examples is:

1. express the discrete Lie-group dynamics using polynomial matrix variables;
2. impose manifold, dynamics, boundary, and input constraints;
3. exploit temporal or user-defined correlative sparsity;
4. generate and solve a sparse Moment--SOS relaxation with SPOT and MOSEK;
5. extract a candidate trajectory from the moment matrices;
6. evaluate rank diagnostics and verify the original polynomial constraints;
7. when available, compare the SDP lower bound with the cost of a verified feasible trajectory.

The Acrobot MPC implementation repeatedly solves the reduced prediction model, applies the extracted first control input, and propagates the full numerical plant with the constrained LGVI. This receding-horizon procedure is intended as a practical controller; it does not by itself certify the complete closed-loop trajectory.

## Citation

If you use the SDP implementation, please cite SPOT and its associated publication:

```bibtex
@inproceedings{kang2025global,
  title     = {Global Contact-Rich Planning with Sparsity-Rich Semidefinite Relaxations},
  author    = {Kang, Shucheng and Liu, Guorui and Yang, Heng},
  booktitle = {Robotics: Science and Systems},
  year      = {2025},
  doi       = {10.15607/RSS.2025.XXI.046}
}
```

SPOT repository: <https://github.com/ComputationalRobotics/SPOT>

## Reproducibility notes

- Optimization results depend on the SPOT, MOSEK, and Python-package versions.
- Verify all reported equality and inequality residuals before treating an extracted trajectory as feasible.
- Rank metrics are numerical tightness diagnostics and are not, by themselves, feasibility or global-optimality certificates.
- Some simulations are intentionally long-horizon experiments and may require substantial runtime and memory.

## Author

Edonis Elshani  
Technical University of Munich and Harvard University
