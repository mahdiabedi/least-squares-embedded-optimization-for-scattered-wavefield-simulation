# least-squares-embedded-optimization-for-scattered-wavefield-simulation
This repository contains the official implementation of our paper:

[Least-Squares-Embedded Optimization for Accelerated Convergence of PINNs in Acoustic Wavefield Simulations](https://arxiv.org/abs/2504.16553)  
Mohammad Mahdi Abedi, David Pardo, Tariq Alkhalifah

---

This work proposes a hybrid training strategy that embeds a least-squares optimization step within the training loop of physics-informed neural networks (PINNs) to solve the 2D acoustic Helmholtz equation more efficiently and in fewer epochs.

Key Features:
- Least-Squares-enhanced training for PINNs
- Forward or Backward differentiation strategy
- Inclusion of perfectly matched layer (PML)
- Inclusion of positional encoder layer
- Varying collocation point strategy
- Comparison against traditional PINNs and finite-difference
- Application to simple and complex velocity models

---
