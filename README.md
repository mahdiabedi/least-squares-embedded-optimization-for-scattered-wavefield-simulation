# least-squares-embedded-optimization-for-scattered-wavefield-simulation
This repository contains the official implementation of the paper "Least-Squares-Embedded Optimization for Accelerated Convergence of PINNs in High-Frequency Acoustic Wavefield Simulations" by Mohammad Mahdi Abedi, David Pardo, Tariq Alkhalifah, published in _Computers and Geosciences_ (2026).
[Least-Squares-Embedded Optimization for Accelerated Convergence of PINNs in Acoustic Wavefield Simulations](https://arxiv.org/abs/2504.16553)  
M M Abedi

---
<img width="300" height="200" alt="Marmousi_velocity" src="https://github.com/user-attachments/assets/2510e113-837d-4a57-aadb-433113125f31" >
<img width="300" height="200" alt="Marmousi30Hz_prediction_LSPINN" src="https://github.com/user-attachments/assets/2cf37ee8-e350-440f-b675-a5e5a6e128b9" >
<img width="300" height="200" alt="Marmousi_30Hz_errors" src="https://github.com/user-attachments/assets/1f7661cc-7e3b-4844-9033-4bf734b43344" >


*(Caption: **Left:** The  Marmousi velocity model. **Center:** The 30Hz scattered acoustic wavefield predicted by our proposed LS-GD method. **Right:** Evolution of errors during training, highlighting the accelerated convergence and stability of LS-GD compared to standard Gradient Descent.)*

## 📄 Abstract

As illustrated in the performance summary above, standard training of PINNs using conventional gradient descent (GD) often suffers from slow convergence and instability when applied to high-frequency seismic wavefields. 

To overcome this in scattered acoustic wavefield simulations based on the Helmholtz equation, we derive a hybrid optimization framework by embedding a least-squares (LS) solver directly into the GD loss function. This approach enables optimal updates for the linear output layer and mitigates the spectral bias of neural networks, making it especially effective for high-frequency problems like the 30Hz Marmousi prediction shown above. We provide practical, tensor-based implementations for Helmholtz formulations with and without perfectly matched layers (PML).

Key Features:
- Least-Squares-enhanced training for PINNs
- Forward or Backward differentiation strategy
- Inclusion of perfectly matched layer (PML)
- Inclusion of positional encoder layer
- Varying collocation point strategy
- Comparison against traditional PINNs and finite-difference
- Application to simple and complex velocity models

---

## Repository Structure

**Source Code:**
- inference_LSGD_PINN.py: The inference script for loading pretrained models and plotting the predicted wavefields on validations points.
- LS_embeded_PINN.py: The primary training script. Allows users to train models using standard Gradient Descent (GD) or the proposed Least-Squares Gradient Descent (LS-GD) for the Simple or Marmousi velocity models.
- My_utilities_LS.py: The  backend containing custom Keras layers, loss function definitions, interpolation utilities, and wavefield plotting routines.

**Data & Pretrained Models:**
- pretrained_models/: Directory containing saved keras models trained with various configurations (tests in the paper).
- FD_results_10Hz_val_velocity.mat: Finite Difference ground truth data for validation (10Hz).
- Simple_random_training.mat and Simple_random_training_PML.mat: Randomly sampled collocation points for training the Simple velocity model (with and without PML).
- xz_Simple_val.npz and xz_Marmousi_val.npz: Validation coordinate grids and velocity profiles for the respective geologic models (used in the inference code).

## Usage Guide
1. Training a Model (LS_embeded_PINN.py)
To train a new model, open LS_embeded_PINN.py and configure the Simulation Settings block at the top of the script. Key parameters include:
- velocity_model: Set to 'simple' or 'marmousi'.
- use_LS: Set to True to enable the proposed Least-Squares embedded solver, or False for conventional Gradient Descent.
- use_PML: Set to True to apply Perfectly Matched Layers at the domain boundaries.
- frequency: Target wavefield frequency.

3. Running Inference & Plotting (inference_LSGD_PINN.py)
To reproduce the plots from the paper using our provided models, use the inference script. 
The User-defined parameters in inference_LSGD_PINN.py are:
- velocity_model = 'Simple'  # 'Simple' or 'Marmousi'
- method = 'LSGD'            # 'GD' or 'LSGD'
- K_max_encoder = 3          # Positional encoder parameter

## Installation & Requirements
The code is written in Python and utilizes **TensorFlow** for automatic differentiation and neural network optimization.

**Install dependencies:**
pip install tensorflow numpy scipy matplotlib keras

