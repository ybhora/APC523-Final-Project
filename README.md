# APC 523 Final Project: Numerical Analysis of Thawing Quintessence

This repository contains the code and analysis for the final project of APC 523 (Numerical Algorithms for Scientific Computing). 

The project is a computational reproduction and numerical analysis of the cosmological dynamical system described in the 2024 paper by Shlivko and Steinhardt, *"Assessing observational constraints on dark energy"*. We computationally simulate the evolution of "thawing quintessence" (a scalar field model for dark energy) by solving a system of nonlinear ODEs, and then evaluate the reliability and intrinsic mathematical degeneracies of fitting this exact physical model to the standard two-parameter empirical CPL model: $w(z) = w_0 + w_a \frac{z}{1+z}$.

## Repository Structure

```text
.
├── data/                  # Directory for generated output arrays (.npz files)
├── figures/               # Directory for generated plots and heatmaps
├── slurm_files/           # Slurm scripts for running jobs on the Adroit cluster
├── analysis.ipynb         # Jupyter notebook for final analysis and visualization
├── convergence.py         # Script to test RK4 global truncation error scaling
├── diagnostics.py         # Script to map the error topology and parameter degeneracy
├── fits.py                # Script for global-local optimization of (w0, wa) parameters
└── solver.py              # Core ODE solver for the quintessence scalar field dynamics
