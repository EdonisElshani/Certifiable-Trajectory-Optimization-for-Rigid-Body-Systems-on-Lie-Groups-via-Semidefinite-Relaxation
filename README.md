# Certifiable Trajectory Optimization for Rigid-Body Systems on Lie Groups via Semidefinite Relaxation

[Master's thesis](Edonis_Elshani_Master_Thesis_Final.pdf) | [Presentation slides](Master_Presentation_Slides.pdf)

Rigid-body trajectory planning is nonconvex because rotations and nonlinear dynamics must hold throughout a motion. This repository studies structure-preserving polynomial models and sparse semidefinite relaxations that can certify an optimal trajectory when the relaxation is tight.

## Method

### Lie-group dynamics

The benchmarks are a controlled 3D pendulum with orientation in $\mathrm{SO}(3)$ and an underactuated two-link Acrobot with link orientations in $\mathrm{SO}(2)\times\mathrm{SO}(2)$. Lie group variational integrators (LGVIs) discretize the mechanics while evolving on the rotation groups. For the Acrobot, maximal coordinates represent each body separately, and the joint constraints eliminate translational positions from the optimization model.

| 3D pendulum ([PDF](pics/3d_pendulum_so3.pdf)) | Acrobot ([PDF](pics/acrobot_so2.pdf)) |
| :---: | :---: |
| ![3D pendulum model](pics/3d_pendulum_so3.png) | ![Acrobot model](pics/acrobot_so2.png) |

### Polynomial optimization problems

The following compact formulations use the implementation's cost weights. They summarize the full indexed constraints in thesis Eqs. (5.46) and (5.56); $R^\star$ and $F^\star$ denote target attitude and relative rotation.

**3D pendulum.** With $z_P=\{R_k,F_k,\bar u_{p,k}\}$, the objective tracks the target rotations while penalizing normalized control:

$$
\begin{aligned}
p_P^\star=\min_{z_P}\quad&
\rho_R\|R_N-R^\star\|_F^2+\rho_F\|F_{N-1}-F^\star\|_F^2\\
&+\alpha_R\sum_{k=0}^{N-1}\|R_k-R^\star\|_F^2
+\alpha_F\sum_{k=0}^{N-2}\|F_k-F^\star\|_F^2\\
&+\frac{1}{\gamma}\sum_{k=1}^{N-1}\|\bar u_{p,k}\|_2^2 .
\end{aligned}
$$

Its quadratic constraints encode the forced LGVI, group kinematics, rotation geometry, and input limits:

$$
\begin{aligned}
F_{k+1}J_d-J_dF_{k+1}^{\top}-J_dF_k+F_k^{\top}J_d
&=h^2(\tau_{g,k+1}+u_{k+1})^\wedge,\\
R_{k+1}&=R_kF_k,\qquad R_k,F_k\in\mathrm{SO}(3),\\
\frac{\operatorname{tr}(F_k)-1}{2}&\geq\cos(\Delta\theta_{\max}),\qquad
\|\bar u_{p,k}\|_2^2\leq 1,\\
R_0&=R^{\mathrm{init}},\qquad F_0=F^{\mathrm{init}} .
\end{aligned}
$$

Here $u_k=(R_k^\top e_3)\times(u_{\max}\bar u_{p,k})$ and $J_d$ is the nonstandard inertia matrix. In the polynomial implementation, $\mathrm{SO}(3)$ is imposed through quadratic column orthogonality and cross-product constraints.

**Acrobot.** For $i\in\{1,2\}$, the reduced decision vector $z_A$ contains $R_{i,k}$, $F_{i,k}$, normalized joint multipliers, and normalized torque $\bar u_k$. Its objective is

$$
\begin{aligned}
p_A^\star=\min_{z_A}\quad&
\rho_R\sum_{i=1}^{2}\|R_{i,N}-R_i^\star\|_F^2
+\rho_F\sum_{i=1}^{2}\|F_{i,N-1}-F_i^\star\|_F^2\\
&+\alpha_R\sum_{k=0}^{N-1}\sum_{i=1}^{2}\|R_{i,k}-R_i^\star\|_F^2\\
&+\alpha_F\sum_{k=0}^{N-2}\sum_{i=1}^{2}\|F_{i,k}-F_i^\star\|_F^2
+\frac{1}{\gamma}\sum_{k=1}^{N-1}\bar u_k^2 .
\end{aligned}
$$

The POP imposes the reduced translational and rotational LGVI equations (thesis Eqs. 5.56b-e), together with

$$
\begin{aligned}
R_{i,k+1}&=R_{i,k}F_{i,k},\qquad R_{i,k},F_{i,k}\in\mathrm{SO}(2),\\
a_{i,k}&\geq\cos(\Delta\theta_{i,\max}),\qquad |\bar u_k|\leq 1,\\
|\bar\lambda_{0,k}^{(j)}|,\ |\bar\lambda_{12,k}^{(j)}|&\leq 1,\qquad j\in\{1,2\},\\
F_{i,0}&=F_i^{\mathrm{init}},\qquad R_{i,1}=R_i^{\mathrm{init}} .
\end{aligned}
$$

Here $a_{i,k}$ is the cosine entry of $F_{i,k}$. The $\mathrm{SO}(2)$ constraints are the quadratic unit-circle equations for the cosine-sine entries. Reconstructing link positions from the joint geometry reduces the full model from $17N+3$ to $13N-1$ scalar optimization variables.

### Sparse relaxation and feedback

[SPOT](https://github.com/ComputationalRobotics/SPOT) converts these POPs into sparse second-order Moment-SOS semidefinite relaxations, solved here with MOSEK. The pendulum uses an automatic minimum-degree sparsity pattern; the Acrobot uses problem-specific cliques that separate dynamics from kinematics. The SDP gives a lower bound. A global certificate additionally needs a feasible extracted trajectory whose cost matches that bound; a nearly rank-one moment matrix alone is only a diagnostic.

For difficult Acrobot swing-ups, SDP-based model predictive control (MPC) extracts the first control, advances the full numerical Acrobot with an LGVI, and replans from the simulated state:

![Acrobot SDP-MPC feedback loop](pics/MPC.png)

[MPC diagram as PDF](pics/MPC.pdf)

## Results

### 3D pendulum

The LGVI simulations preserve rotation geometry and show better long-horizon energy behavior than the compared Runge-Kutta simulations. In optimization, all five evaluated target rotations yield feasible, numerically rank-one extractions whose costs agree with the SDP lower bounds. The 180 deg trajectory below is therefore a certified offline result; its reported solve took about 64.6 minutes.

<img src="pics/pendulum_180deg.gif" alt="3D pendulum 180 degree trajectory" width="420">

### Acrobot MPC

The empirical basins sample $10\times10$ starts from rest, with both initial link angles in $\{0,20,\ldots,180\}$ deg and a 15 Nm torque bound. Stabilization requires attitude and step-rotation errors below 2 deg for three consecutive updates. Matched prediction, simulation, and control steps stabilize 94/100 starts in Cases I and II. The mismatched Case III stabilizes 34/100. These are simulated closed-loop outcomes, not global-optimality certificates.

| | Case I | Case II | Case III |
| :--- | :---: | :---: | :---: |
| Prediction / simulation / control step (s) | 0.1 / 0.1 / 0.1 | 0.05 / 0.05 / 0.05 | 0.1 / 0.0025 / 0.025 |
| Prediction steps $N$ | 20 | 40 | 20 |
| Stabilized starts | **94/100** | **94/100** | **34/100** |
| Basin | ![Case I basin of attraction](pics/basin_case_I.png) | ![Case II basin of attraction](pics/basin_case_II.png) | ![Case III basin of attraction](pics/basin_case_III.png) |

Vector versions: [Case I](pics/basin_case_I.pdf) | [Case II](pics/basin_case_II.pdf) | [Case III](pics/basin_case_III.pdf)

| Acrobot stabilization (Case I) | Acrobot stabilization (Case III) |
| :---: | :---: |
| ![Acrobot 180 deg swing-up](pics/acrobot_180deg.gif) | ![Second Acrobot 180 deg swing-up](pics/acrobot_180deg_25.gif) |

## Running the code

### Numerical simulations

The `Numerical_Simulation/` folder contains standalone LGVI and Runge-Kutta comparisons. Install NumPy, SciPy, Matplotlib, PyYAML, and Pillow, then run the entry points from this repository's root:

```bash
python -m pip install numpy scipy matplotlib pyyaml pillow
python Numerical_Simulation/main_numerical_simulation.py
python Numerical_Simulation/Acrobot/main_acrobot_unforced.py
python Numerical_Simulation/Acrobot/main_acrobot_forced_sine.py
```

The 3D pendulum script displays comparison plots. The Acrobot scripts save plots under `Numerical_Simulation/Acrobot/output/`. Their default horizons are long; adjust `TF` or `tf` in the corresponding entry point for a shorter experiment.

### SDP optimization and MPC

Install and build the [SPOT Python interface](https://github.com/ComputationalRobotics/SPOT#-python-installation-and-usage) first, including its MOSEK solver dependency. This repository does not vendor SPOT. The optimization imports assume the **following exact placement** inside a SPOT clone:

```text
SPOT/
|-- SPOT/PYTHON/                         # SPOT package and compiled binding
|-- Python-examples/
|   |-- pendulum_SO3_Python.py            # copy from Ridged_Body_SDP/
|   |-- acrobot_SO2_python/              # copy the whole directory
|       |-- open_loop_sdp.py
|       |-- main_sdp_mpc.py
|       |-- config/acrobot_physical.yaml
|       |-- SDP/
|       |-- Numerical_Simulation/
```

For example, if this repository and `SPOT` are sibling folders, copy the files from this repository root in PowerShell:

```powershell
Copy-Item .\Ridged_Body_SDP\pendulum_SO3_Python.py ..\SPOT\Python-examples\
Copy-Item .\Ridged_Body_SDP\acrobot_SO2_python ..\SPOT\Python-examples\ -Recurse
```

Then run from the **SPOT repository root**:

```bash
python Python-examples/pendulum_SO3_Python.py
python Python-examples/acrobot_SO2_python/open_loop_sdp.py
python Python-examples/acrobot_SO2_python/main_sdp_mpc.py
```

Set physical parameters, initial/target angles, `dt_sdp`, `dt_sim`, `control_interval`, horizon, and bounds in `Python-examples/acrobot_SO2_python/config/acrobot_physical.yaml`. The supplied file has Case II time scales ($0.05$ s, $N=40$) but runs **one** initial condition per invocation; the basin plots summarize separate 100-start evaluations. Acrobot SDP and MPC outputs are written under `Python-examples/acrobot_SO2_python/Results/`. These solves can take hours.

SPOT reference: Shucheng Kang, Guorui Liu, and Heng Yang, "[Global Contact-Rich Planning with Sparsity-Rich Semidefinite Relaxations](https://www.roboticsproceedings.org/rss21/p046.html)," *Robotics: Science and Systems*, 2025. DOI: [10.15607/RSS.2025.XXI.046](https://doi.org/10.15607/RSS.2025.XXI.046).
