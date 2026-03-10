#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference and Visualization Suite
---------------------------------
Research Paper: "Least-Squares-Embedded Optimization for Accelerated Convergence 
                 of PINNs in High-Frequency Acoustic Wavefield Simulations"
Journal: Computers and Geosciences


Description:
    This script performs inference using pre-trained PINN models (GD or LS-GD).
    It supports the test on positional encoding with flexible frequency bands (K_max).
    
 M.M. Abedi 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# =============================================================================
# 1. USER-DEFINED PARAMETERS
# =============================================================================

# Select the geologic scenario: 'Simple' (Figure 2) or 'Marmousi' (Figure 6)
velocity_model = 'Simple' 

# Select optimization method: 
# 'GD'   = Conventional Gradient Descent
# 'LSGD' = Proposed Least-Squares-Embedded Gradient Descent
method = 'LSGD' 

# Multi-scale Encoder Settings:
# For 'Simple': select K_max_encoder from [0, 3, 5]
K_max_encoder = 3 
if velocity_model == 'Marmousi':# For 'Marmousi': fixed at 5 
    K_max_encoder = 5


#%% loading
# Path to the specific pre-trained model file
model_path = f"PreTrained_Models/{velocity_model}_{method}_K{K_max_encoder}.keras"


# Load validation coordinates and corresponding velocity model
# xz_val: [N, 2] array of spatial coordinates (x, z)
# v_val:  [N, 1] array of acoustic velocity values
data = np.load(f'xz_{velocity_model}_val.npz')

v_val        = data['v_val']
xz_val       = data['xz_val']
npts_x_val   = int(data['npts_x_val'])
npts_z_val   = int(data['npts_z_val'])
extent       = data['extent_original']  # Format: [x_min, x_max, z_min, z_max]
 #%% functions   
plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for text rendering
    "font.family": "serif",        # Set font family to serif
    "font.serif": ["Times"],       # Use Times as the serif font
    "font.size": 14,               # Set the default font size
    "axes.titlesize": 19,          # Title font size
    "axes.labelsize": 16,          # Label font size
    "xtick.labelsize": 14,         # x-tick label font size
    "ytick.labelsize": 14,         # y-tick label font size
    "text.latex.preamble": r"\usepackage{amsmath}"  # Use amsmath for better LaTeX rendering
})

def sin_activation(x):
    return tf.sin(x)

def to_complex_grid(U_ri, nz, nx):
    """
    U_ri: Tensor or array shape (N,2) with N = nz*nx, order row-major (z major).
    Returns: complex tensor shape (nz, nx) with axis0 = z (rows), axis1 = x (cols).
    """
    U_ri = tf.convert_to_tensor(U_ri)
    Uc = tf.complex(U_ri[:,0], U_ri[:,1])   # flat complex (N,)
    return tf.reshape(Uc, [nz, nx])        # row-major reshape


def get_custom_embedder(max_freq):
    """
    Returns an EmbedderLayer class configured for a specific maximum frequency.
    """
    class EmbedderLayer(tf.keras.layers.Layer):
        def __init__(self, domain_bounds, **kwargs):
            super(EmbedderLayer, self).__init__(**kwargs)
            self.domain_bounds = domain_bounds
            
            # If max_freq is 5, range(5 + 1) gives [0, 1, 2, 3, 4, 5]
            # self.freq_bands becomes [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
            self.freq_bands = [2.0 ** i for i in range(max_freq + 1)]

        @tf.function()
        def call(self, inputs):
            # Multiply inputs by each frequency band dynamically
            scaled_inputs = [tf.math.multiply(inputs, freq) for freq in self.freq_bands]
            
            # Concatenate all scaled inputs
            input_all = tf.concat(scaled_inputs, axis=1)
            
            # Apply sine and cosine functions
            sin_embed = tf.sin(input_all)
            cos_embed = tf.cos(input_all)

            # Concatenate original input, sine, and cosine embeddings
            output = tf.concat([inputs, sin_embed, cos_embed], axis=1)
            return output

        def get_config(self):
            config = super(EmbedderLayer, self).get_config()
            config.update({"domain_bounds": self.domain_bounds})
            return config
            
    return EmbedderLayer

def model_prediction(model_path,x_in):
    u_model =keras.models.load_model(model_path,
        custom_objects={
            'EmbedderLayer': get_custom_embedder(max_freq=K_max_encoder),
            'sin_activation': sin_activation}, compile=False)

    prediction = u_model(x_in)
    prediction_complex=to_complex_grid(prediction, npts_z_val, npts_x_val)
    
    # Extract real and imaginary parts
    u_real = prediction[:, 0]  # Real part
    u_real_grid = tf.reshape(u_real, (npts_z_val, npts_x_val)).numpy()
    u_imag = prediction[:, 1]  # imag part
    u_imag_grid = tf.reshape(u_imag, (npts_z_val, npts_x_val)).numpy()
    return u_real_grid,u_imag_grid,prediction_complex


#%% plottings
#Velocity model
plt.figure(figsize=(6,4.5))
plt.imshow(tf.reshape(v_val, (npts_z_val, npts_x_val)), extent=extent, origin="upper", cmap="viridis", aspect="auto")
plt.title(f"{velocity_model} Velocity model")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
cbar = plt.colorbar(label='$v$ (normalized)', orientation='vertical')
cbar.ax.invert_yaxis()
plt.tight_layout()  
plt.show()

#Model prediction:
u_real_prediction,u_imag_prediction,prediction_complex=model_prediction(model_path,xz_val)

plt.figure(figsize=(12,5))
plt.suptitle(f"{method} Prediction for {velocity_model} model", fontsize=21, y=.98)
plt.subplot(1, 2, 1)
plt.imshow(u_real_prediction, extent=extent, origin="upper",
           cmap="seismic", aspect="auto")
plt.title("Real part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(u_imag_prediction, extent=extent, origin="upper",
           cmap="seismic", aspect="auto")
plt.title("Imaginary part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()
plt.tight_layout()
plt.show()
