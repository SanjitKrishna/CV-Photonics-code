# Photonic Loop Memory for Continuous-Variable Quantum Communication: simulation code

Simulation code and figures for the paper

> **Photonic Loop Memory with Logged-Displacement Rollback and Digital-Twin Calibration for Continuous-Variable Quantum Communication**
> Sanjit Krishna and Himanshu Pandey, Department of Physics, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India.

A preliminary version of this work: S. Krishna, *A Reversible Continuous-Variable Photonic Memory Architecture Based on Displacement-Evolved Coherent-State Dynamics*, [arXiv:2607.18702](https://arxiv.org/abs/2607.18702).

The script `cv_final_simulations.py` regenerates every figure of the paper and prints the numbers quoted in the text.

## What is modelled

A time-multiplexed fiber loop holds M = 8 coherent-state pulses (1 ns apart, so one lap = 8 ns and the loop is 1.6 m of fiber). Each lap the amplitude evolves as

    alpha_{n+1} = lambda * alpha_n + Delta_n + eps_n,    lambda = sqrt(eta * G)

with loss, an optional phase-insensitive amplifier and Gaussian noise. The script covers:

* quantum-limited fidelity of an amplified loop, `F_n = 1 / (1 + n(1-eta)/eta)`;
* link budget: lifetime, delay and equivalent fiber length;
* rollback of logged displacements: Monte Carlo and the closed-form variance;
* delayed feed-forward and a comparison with a fixed fiber spool and a switched delay bank;
* a squeezed (entangled) payload: entanglement lifetime, logarithmic negativity and teleportation fidelity.

## Install and run

```bash
pip install numpy matplotlib          # or: pip install -r requirements.txt
python cv_final_simulations.py                  # figures -> ./figures
python cv_final_simulations.py --out Images     # figures -> ./Images
```

Python 3.8 or newer. Only NumPy and Matplotlib are needed.

## Figure map

| Paper figure | Output file | Function |
|---|---|---|
| Fig. 2 forward fidelity | `Figure_fwd_quantum_limited.png` | `fig_forward` |
| Fig. 3 link budget | `Figure_comm_budget.png` | `fig_comm` |
| Fig. 4 feed-forward | `Figure_feedforward.png` | `fig_ff` |
| Fig. 5 / Table IV delay-line comparison | `Figure_delay_comparison.png` | `fig_delaycomp`, `delay_comparison` |
| Fig. 6 amplitude trajectories | `Figure_trajectories.png` | `fig_trajectories` |
| Fig. 7 rollback fidelity (Monte Carlo) | `Figure_rollback_zoom.png` | `fig_rollback` |
| Fig. 8 rollback fidelity (analytic) | `Figure_analytic_stability.png` | `fig_analytic`, `rollback_var` |
| Fig. 9 squeezed payload | `Figure_entanglement.png` | `fig_entanglement` |

## Reproducibility

* Random numbers use a fixed seed (`SEED = 2026`). Monte Carlo curves average `R = 2000` realizations. The seed is arbitrary; another seed changes the curves only within Monte Carlo error.
* The functions are called in a fixed order in the `__main__` block because the random numbers are drawn in sequence. Keep that order if you want identical numbers.
* The trajectory figure uses its own generator, so it does not affect the others.
* Example printed result: rollback fidelity at depth 50 is 0.9892 (Monte Carlo) against 0.9894 (formula).

## Parameters and assumptions

Constants are at the top of the script. Some values are **assumptions chosen to illustrate the formulas, not measured values**: switch insertion loss 0.1 dB, feed-forward latency window 0.1 to 0.5 us, rollback process noise and twin estimation noise 0.008, and the trajectory parameters (lambda = 0.997, noise 0.02). Change them to test other cases.

The model is an amplitude-level description with Gaussian noise, so the results test the mathematics of the model and are not experimental data. The entanglement results assume ideal squeezing, a lossless partner mode and phase-insensitive noise.

## Files

```
cv_final_simulations.py   all simulations and figures
requirements.txt          numpy, matplotlib
README.md                 this file
```

## Citation

If you use this code, please cite the paper and the preprint above.

## Contact

Sanjit Krishna, i24ph006@phy.svnit.ac.in
