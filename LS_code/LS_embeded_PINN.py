#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LS PINN training.

Created on Sun Apr 2 2025
Author: Mohammad Mahdi Abedi
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import keras

from My_utilities_LS import (
    make_loss, make_loss_PML, LS_diff, LS_diff_PML, sin_activation,
    create_xz_reg, compute_U0, interpolator, plot_model_wavefield,
    save_model_and_history
)

# Set default TensorFlow float type
tf.keras.backend.set_floatx("float32")
dtype = "float32"

# ------------------------#
#   Simulation Settings   #
# ------------------------#

# General configuration
frequency = 10                      # Frequency in Hz
neurons = 16                        # Hidden layer neuron count
neurons_final = 16                  # Penultimate layer neuron count
activation = sin_activation         # Activation function for hidden layers
activation_penultima = activation   # Activation for penultimate layer

learning_rate = 0.002
num_epochs = 30000
          
velocity_model = 'simple'          # Options: 'simple', 'overthrust', 'marmousi'

use_LS = True                       # Use least-squares solver at final layer
use_PML = False                     # Use Perfectly Matched Layer (PML)
use_source_reg = True              # Use source regularization (soft constraint)
use_lr_decay = True                # Apply learning rate decay
varying_colocation = True          # Change collocation points per epoch
GD_loss="Backward"                 # Two options for calculation of gradient descent loss: "Backward", "Forward" 
plotting = False                   # Plot results after training
saving = False                     # Save model and results

# Regularization and initialization
beta = 15.85                        # Soft constraint penalty
lamda = 1.0                         # Regularization parameter
seed = 1234                         # Random seed

# -----------------------------#
#   Output Directory           #
# -----------------------------#

if saving:
    os.system('rm -rfv Results/*')
    os.makedirs('Results/Models')

# ----------------------------#
#   Domain Configuration      #
# ----------------------------#

# Define spatial domain for the velocity model
if velocity_model == 'simple':
    a_x, b_x = 0.0, 2.5  # x-axis bounds [km]
    a_z, b_z = 0.0, 2.5  # z-axis bounds [km]

    if frequency == 10:
        npts_x = 31      # Grid points in x-direction
        npts_z = 31      # Grid points in z-direction

domain_bounds = (a_x, b_x, a_z, b_z)
domain_bounds_valid = domain_bounds

# Compute angular frequency
omega = np.float32(frequency * 2 * np.pi)

# ----------------------------#
#   PML Configuration         #
# ----------------------------#

if use_PML:
    L_PML = 0.5          # Thickness of the PML layer
    a0 = 0.8             # Absorbing strength
    omega0 = omega       # Reference angular frequency

    # Coupling constant for PML formulation
    c = tf.cast((a0 * omega0) / (omega * L_PML**2),dtype)

    # Adjust domain and grid size for PML extension
    xz_PML = (a_x, b_x, a_z, b_z)
    npts_x = int(npts_x * ((b_x - a_x) + 2 * L_PML) / (b_x - a_x))
    npts_z = int(npts_z * ((b_z - a_z) + 2 * L_PML) / (b_z - a_z))

    # Extended domain bounds
    domain_bounds_PML = np.add(domain_bounds, [-L_PML, L_PML, -L_PML, L_PML])

# ----------------------------#
#   Least-Squares Parameters  #
# ----------------------------#

initial_reg = 0.1          # Initial LS regularization strength
reg_decay_rate = 0.2       # Rate of decay for regularization
final_reg = 1e-4           # Final LS regularization value

# Total number of collocation points per epoch
npts = npts_x * npts_z

#%% Loading the validation data

#loading the validation finite-difference result and exact source and coordinates
if velocity_model=='overthrust':
    mat_data = scipy.io.loadmat(f'FD_results_{frequency}Hz_val_OVE_velocity_v0corrected.mat')#Hz_val_velocity, Hz_val_OVE_velocity
elif velocity_model=='simple':
    mat_data = scipy.io.loadmat(f'FD_results_{frequency}Hz_val_velocity.mat')#Hz_val_velocity, Hz_val_OVE_velocity
elif velocity_model=='marmousi':
    mat_data = scipy.io.loadmat(f'FD_results_{frequency}Hz_val_velocity_marmousi.mat')
    
val_keys = list(mat_data.keys())
U0_analytic=mat_data['U0_analytic']
U0_fd=mat_data['U0_2d']
dU_2d=mat_data['dU_2d']
xz_val=tf.reshape(mat_data['xz_val'],(-1,2))
v_val=mat_data['v_val']
v_val=tf.reshape(v_val,(-1,1))
[npts_z_val,npts_x_val]=np.shape(dU_2d)

s_x=np.float32(np.squeeze(mat_data['s_x']))
s_z=np.float32(np.squeeze(mat_data['s_z']))
s_xz=tf.cast(tf.stack([s_x,s_z],axis=0),dtype=dtype)
factor=np.float32(tf.squeeze(mat_data['factor']))
U0_analytic=tf.concat([tf.reshape(tf.math.real(U0_analytic),(-1, 1)),tf.reshape(tf.math.imag(U0_analytic),(-1, 1))], axis=1)
# dU_2d=tf.concat([tf.reshape(tf.math.real(dU_2d),(-1, 1)),tf.reshape(tf.math.imag(dU_2d),(-1, 1))], axis=1)
dU_2d_r=interpolator(np.real(dU_2d),domain_bounds_valid,xz_val,dtype=tf.float32) 
dU_2d_i=interpolator(np.imag(dU_2d),domain_bounds_valid,xz_val,dtype=tf.float32)
dU_2d=tf.concat([tf.reshape(dU_2d_r,(-1, 1)),tf.reshape(dU_2d_i,(-1, 1))], axis=1)

v0=np.float32(np.squeeze(mat_data['v0']))

#%% Loading random colocation points:
if varying_colocation:
    if velocity_model=='simple':
        if not use_PML:

            mat_data = scipy.io.loadmat('Simple_random_training.mat')
            v_all=mat_data['v_all']
            xz_all=mat_data['xz_all']
    
        if use_PML:
            print('PML')
            mat_data = scipy.io.loadmat('Simple_random_training_PML.mat')
            v_all=mat_data['v_all']
            xz_all=mat_data['xz_all']
 
    n_all=xz_all.shape[0]


#%%Model buildng
def make_u_model(neurons, activation=tf.math.sin, activation_penultima=tf.math.sin, neurons_final=None, dtype=tf.float32, trainableLastLayer=False,v0=1., omega=1. ,seed=1234):
    # Xavier (Glorot) initialization is commonly used for PINNs
    kernel_regularizer =keras.regularizers.L2(l2=0)
    # Use GlorotNormal (Xavier) initializer for the kernel
    b_init =keras.initializers.Zeros()  # Use zero bias initialization

    if neurons_final is None:
        neurons_final = neurons

    # Input layer
    l0 =keras.layers.Input(shape=(2,), name="x_input", dtype=dtype)
    
    # Apply the embedding layer
    l1 = EmbedderLayer(name="embedder",domain_bounds = domain_bounds )(l0)
    
    # First dense layer 
    l1 =keras.layers.Dense(neurons, activation=activation, dtype=dtype,
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.GlorotNormal(seed=seed),  # Different seed per layer
                            bias_initializer=b_init, name="layer_1")(l1)
    
    # Second dense layer 
    l1 =keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_2",
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.GlorotNormal(seed=seed+1),  # Different seed
                            bias_initializer=b_init)(l1)
    
    # # Third dense layer 
    l1 =keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_3",
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.GlorotNormal(seed=seed+2),  # Different seed
                            bias_initializer=b_init)(l1)
    
    # # Fourth dense layer 
    # l1 =keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_4",
    #                         kernel_regularizer=kernel_regularizer,
    #                         kernel_initializer=keras.initializers.GlorotNormal(seed=seed+5),  # Different seed
    #                         bias_initializer=b_init)(l1)
    

    l1 =keras.layers.Dense(neurons_final, activation=activation_penultima, dtype=dtype, name="penultimate_layer",
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.GlorotNormal(seed=seed+3),  # Different seed
                            bias_initializer=b_init)(l1)

    output =keras.layers.Dense(2, use_bias=False, trainable=trainableLastLayer, dtype=dtype, name='Output_layer',
                            kernel_initializer=keras.initializers.GlorotNormal(seed=seed+4))(l1)  # Different seed

    # # Define models
    u_bases =keras.Model(inputs=l0, outputs=l1)
    u_model =keras.Model(inputs=l0, outputs=output)
    
    return u_model, u_bases

# ###### positional encoding!!!!!!!!!!    
class EmbedderLayer(tf.keras.layers.Layer):#The old embedder
    def __init__(self, domain_bounds, **kwargs):
        super(EmbedderLayer, self).__init__(**kwargs)
        self.domain_bounds = domain_bounds  # Store domain bounds for normalization

    @tf.function()
    def call(self, inputs):

        input1 =  (inputs)
        input2 = tf.math.multiply(input1 , 2.0)
        input4 = tf.math.multiply(input1 , 4.0)
        input8 = tf.math.multiply(input1 , 8.0)

        input_all = tf.concat([input1, input2, input4, input8], axis=1)

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
 
#Two options for training: 
    #1. recalculate the gradient descent loss with back propagation:
@tf.function()
def train_step_withLS(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,U0_val, xz_val, v_val,compute_validation=True,model_type='PINN',
                      use_source_reg=False,source_reg=None,source_reg_val=None,num_reg_points=500,cal_error=False,dU_2d_val=0.,use_Vin=False,l2_regularizer=10**-4):
    reg_loss=0.        
    B,R= LS_diff(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,model_type,use_source_reg,source_reg,num_reg_points=num_reg_points,use_Vin=use_Vin)#matrices of derivatives and right-hand side
    weights_optimal = tf.linalg.lstsq(B, R, l2_regularizer=l2_regularizer)#solving LS
    with tf.GradientTape() as tape:
        u_model.layers[-1].kernel.assign(weights_optimal)
        Loss,reg_loss,_  =  make_loss (u_model, U0, xz, v, v0, omega, lamda,use_source_reg,source_reg,cal_error=False,use_Vin=use_Vin,num_reg_points=num_reg_points)#calculate the GD loss using a backward differentiation

    if  compute_validation:
        Loss_valid,_,error = make_loss(u_model, U0_val, xz_val, v_val, v0, omega, lamda,use_source_reg,source_reg_val,cal_error,dU_2d_val,use_Vin)
    else:
        Loss_valid =0.
        error=0.
    gradients = tape.gradient(Loss, u_model.trainable_variables)#excluding the last layer, even if it is set as trainable
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    return Loss,Loss_valid,reg_loss,error,l2_regularizer

    #2. Here we do not recalculate the derivatives to calculate the GD loss, but sometimes get a warning for graph topological ordering that slows the calculation
@tf.function()
def train_step_withLS2(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,U0_val, xz_val, v_val,compute_validation=True,model_type='PINN',
                      use_source_reg=False,source_reg=None,source_reg_val=None,num_reg_points=500,cal_error=False,dU_2d_val=0.,use_Vin=False,l2_regularizer=10**-4):
    reg_loss=0.
    with tf.GradientTape() as tape:
        B,R= LS_diff(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,model_type,use_source_reg,source_reg,num_reg_points,use_Vin)
        # Solve the least squares problem to find weights of shape [neurons_final, 2]
        weights_optimal = tf.linalg.lstsq(B, R, l2_regularizer=l2_regularizer)# Gradient not defined for fast=False, the topological problem is Not from LS
        
        Loss=tf.matmul(B,weights_optimal)-R#calculating the loss using the already calculated derivatives in B
        Loss=tf.reduce_sum(tf.square(Loss))    
        u_model.layers[-1].kernel.assign(weights_optimal)

    if compute_validation:
        Loss_valid,_,error = make_loss(u_model, U0_val, xz_val, v_val, v0, omega, lamda,use_source_reg,source_reg_val,cal_error,dU_2d_val,use_Vin)
    else:
        Loss_valid =0.
        error=0.
    gradients = tape.gradient(Loss, u_model.trainable_variables)#excluding the last layer, even if it is set as trainable
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    return Loss,Loss_valid,reg_loss,error,l2_regularizer

@tf.function()
def train_step_withLS_PML(u_model, u_bases, U0, xz, v, v0, omega,lamda,npts,c,xz_PML,U0_val, xz_val, v_val,compute_validation=True,model_type='PINN',
                      use_source_reg=False,source_reg=None,source_reg_val=None,num_reg_points=500,cal_error=False,dU_2d_val=0.,use_Vin=False,l2_regularizer=10**-4):
    reg_loss=0.
    B,R= LS_diff_PML(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,model_type,c,xz_PML,use_source_reg,source_reg,num_reg_points,use_Vin)
    # Solve the least squares problem to find weights of shape [neurons_final, 2]
    weights_optimal = tf.linalg.lstsq(B, R, l2_regularizer=l2_regularizer)# Gradient not defined for fast=False
    # Split [wr;wi] into two halves and concatenate them to the shape [[wr wi]]
    wr, wi = tf.split(weights_optimal, num_or_size_splits=2, axis=0)  # Each has shape (n_neurons, 1)
    weights_optimal = tf.concat([wr, wi], axis=1)

    with tf.GradientTape() as tape:
        u_model.layers[-1].kernel.assign(weights_optimal)
        Loss,reg_loss,_ = make_loss_PML  (u_model, U0, xz, v, v0, omega, lamda, omega0, c,xz_PML, use_source_reg, source_reg, cal_error=False)

    if compute_validation:
        Loss_valid,_,error = make_loss(u_model, U0_val, xz_val, v_val, v0, omega, lamda,use_source_reg,source_reg_val,cal_error,dU_2d_val,use_Vin)
    else:
        Loss_valid =0.
        error=0.
    gradients = tape.gradient(Loss, u_model.trainable_variables)#excluding the last layer, even if it is set as trainable
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    return Loss,Loss_valid,reg_loss,error

@tf.function()
def train_step_withLS2_PML(u_model, u_bases, U0, xz, v, v0, omega,lamda,npts,c,xz_PML,U0_val, xz_val, v_val,compute_validation=True,model_type='PINN',
                      use_source_reg=False,source_reg=None,source_reg_val=None,num_reg_points=500,cal_error=False,dU_2d_val=0.,use_Vin=False,l2_regularizer=10**-4):
    reg_loss=0.
    with tf.GradientTape() as tape:
        B,R= LS_diff_PML(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,model_type,c,xz_PML,use_source_reg,source_reg,num_reg_points,use_Vin)
        # Solve the least squares problem to find weights of shape [neurons_final, 2]
        weights_optimal = tf.linalg.lstsq(B, R, l2_regularizer=l2_regularizer)# Gradient not defined for fast=False

        Loss=tf.matmul(B,weights_optimal)-R
        Loss=tf.reduce_sum(tf.square(Loss))    
        # Split [wr;wi] into two halves and concatenate them to the shape [[wr wi]]
        wr, wi = tf.split(weights_optimal, num_or_size_splits=2, axis=0)  # Each has shape (n_neurons, 1)
        weights_optimal = tf.concat([wr, wi], axis=1)
        # tf.print((weights_optimal))
        u_model.layers[-1].kernel.assign(weights_optimal)

    if compute_validation:
        Loss_valid,_,error = make_loss(u_model, U0_val, xz_val, v_val, v0, omega, lamda,use_source_reg,source_reg_val,cal_error,dU_2d_val,use_Vin)
    else:
        Loss_valid =0.
        error=0.
    gradients = tape.gradient(Loss, u_model.trainable_variables)#excluding the last layer, even if it is set as trainable
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    return Loss,Loss_valid,reg_loss,error

# Training step without LS
@tf.function()
def train_step(u_model, U0, xz,v, v0, omega, lamda,U0_val, xz_val, v_val,compute_validation=True,
               use_source_reg=False,source_reg=None,source_reg_val=None,cal_error=False,dU_2d_val=0.,use_Vin=False):
    reg_loss=0.
    with tf.GradientTape() as tape:
        Loss,reg_loss,_ = make_loss      (u_model, U0, xz, v, v0, omega, lamda,use_source_reg,source_reg,cal_error=False,use_Vin=use_Vin)

    if compute_validation:
        Loss_valid,_,error = make_loss   (u_model, U0_val, xz_val, v_val, v0, omega, lamda,use_source_reg,source_reg_val,cal_error,dU_2d_val,use_Vin)
    else:
        Loss_valid =0.
        error=0.
    # Compute the gradients and apply them to the model's weights
    gradients = tape.gradient(Loss, u_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    
    return Loss,Loss_valid,reg_loss,error

# Training step without LS, for PDE that includes PML
@tf.function()
def train_step_PML(u_model, U0, xz,v, v0, omega, lamda, omega0, c,xz_PML,U0_val, xz_val, v_val,compute_validation=True,
               use_source_reg=False,source_reg=None,source_reg_val=None,cal_error=False,dU_2d_val=0.):
    reg_loss=0.
    with tf.GradientTape() as tape:
        Loss,reg_loss,_ = make_loss_PML  (u_model, U0, xz, v, v0, omega, lamda,  omega0, c,xz_PML, use_source_reg, source_reg, cal_error=False)

    if compute_validation:
        Loss_valid,_,error = make_loss_PML(u_model, U0_val, xz_val, v_val, v0, omega, lamda,  omega0, c,xz_PML, use_source_reg,source_reg_val,cal_error,dU_2d_val)
    else:
        Loss_valid =0.
        error=0.
    # Compute the gradients and apply them to the model's weights
    gradients = tape.gradient(Loss, u_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    
    return Loss,Loss_valid,reg_loss,error


# Define the exponential decay function for LS regularization epsilon
def exponential_decay_reg(initial_reg,final_reg, decay_rate, step):
    return initial_reg * tf.exp(-decay_rate * step)+final_reg


if not use_LS:
    initial_reg=0
    reg_decay_rate = 0
    final_reg=0

# Define the optimizer
if use_lr_decay:
    initial_learning_rate=learning_rate
    decay_steps=10000
    decay_rate=0.9
    final_learning_rate = initial_learning_rate * (decay_rate ** (num_epochs / decay_steps))
    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate)
    
if not use_PML:
    a0=0
    L_PML=0
if not use_source_reg:
    beta=0
#define parameters in history
parameters = {
    # "ERROR is MSE for comparison to Alkhalifah":1,

    "velocity_model":velocity_model,
    "use_LS": use_LS,
    "LS_reg":[initial_reg,reg_decay_rate,final_reg],
    "use_source_reg":use_source_reg,
    "use_PML":use_PML,
    "a0":a0,
    "L_PML":L_PML,
    "omega": float(omega),
    "v0": v0,
    "lamda": lamda,
    "neurons": neurons,
    "neurons_final": neurons_final,
    "activation": activation,
    "activation_penultima": activation_penultima,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "varying_colocation": varying_colocation,
    "domain_bounds":domain_bounds,
    "Source_xz": list(s_xz.numpy()),
    "npts_x": npts_x,
    "npts_z": npts_z,
    "seed":seed,
    "beta":beta}   

optimizer =keras.optimizers.Adam(learning_rate=learning_rate)
print(parameters)

u_model, u_bases = make_u_model(neurons,neurons_final=neurons_final,activation=activation,activation_penultima=activation_penultima,
                            trainableLastLayer=not use_LS,v0=v0,omega=omega,seed=seed)

u_model.compile(optimizer=optimizer,loss = make_loss)
u_model.summary()


#%% Training Loop (new Version)
start_time = time.time()
epoch_time = start_time

Loss, Loss_val, Error_val, constraint_Loss = [], [], [], []

U0s = compute_U0(s_xz, s_xz, v0, omega, factor)
U0_all = compute_U0(xz_all, s_xz, v0, omega, factor)
U0_val = compute_U0(xz_val, s_xz, v0, omega, factor)
source_reg_val = create_xz_reg(xz_val, s_xz, omega, v0, beta)

if varying_colocation:
    _, _, indices_around_source = create_xz_reg(xz_all, s_xz, omega, v0, beta, num_reg_points=npts // 100)
else:
    rng = np.random.default_rng(seed=1)
    random_indices = np.sort(rng.choice(n_all, npts, replace=False))
    random_indices = np.concatenate((random_indices, indices_around_source))
    xz_train = tf.gather(xz_all, random_indices)
    v_train = tf.gather(v_all, random_indices)
    U0_train = tf.gather(U0_all, random_indices)

def get_training_data(epoch):
    rng = np.random.default_rng(seed=epoch)
    idx = np.sort(rng.choice(n_all, npts, replace=False))
    idx = np.concatenate((idx, indices_around_source))
    xz = tf.gather(xz_all, idx)
    v = tf.gather(v_all, idx)
    U0 = tf.gather(U0_all, idx)
    source_reg = create_xz_reg(xz, s_xz, omega, v0, beta) if use_source_reg else []
    return xz, v, U0, source_reg




#training loop:
for epoch in range(num_epochs):
    
    cal_error = (epoch % 100 == 0)# True or False, calculation error every 100 epoch
    compute_validation = cal_error

    if varying_colocation:
        xz_train, v_train, U0_train, source_reg = get_training_data(epoch)

    if use_LS:
        l2_regularizer = exponential_decay_reg(initial_reg, final_reg, reg_decay_rate, epoch)

    common_args = dict(
        u_model=u_model,
        U0=U0_train,
        xz=xz_train,
        v=v_train,
        v0=v0,
        omega=omega,
        lamda=lamda,
        U0_val=U0_val,
        xz_val=xz_val,
        v_val=v_val,
        use_source_reg=use_source_reg,
        source_reg_val=source_reg_val,
        dU_2d_val=dU_2d,
        compute_validation=compute_validation,
        source_reg=source_reg,
        cal_error=cal_error
    )                 
    if use_LS:
        if GD_loss == "Backward":
            if not use_PML:
                loss_train, loss_val, reg_loss, error_val, l2_regularizer_adapted = train_step_withLS(
                    **common_args,
                    u_bases=u_bases,
                    npts=npts + npts // 100,
                    num_reg_points=500,
                    l2_regularizer=l2_regularizer
                )
            if use_PML:
                loss_train, loss_val, reg_loss, error_val = train_step_withLS_PML(
                    **common_args,
                    u_bases=u_bases,
                    npts=npts + npts // 100,
                    num_reg_points=500,
                    l2_regularizer=l2_regularizer,
                    c=c,
                    xz_PML=xz_PML
                )
        elif GD_loss == "Forward":
            if not use_PML:
                loss_train, loss_val, reg_loss, error_val, l2_regularizer_adapted = train_step_withLS2(
                    **common_args,
                    u_bases=u_bases,
                    npts=npts + npts // 100,
                    num_reg_points=500,
                    l2_regularizer=l2_regularizer
                )
            if use_PML:
                loss_train, loss_val, reg_loss, error_val = train_step_withLS2_PML(
                    **common_args,
                    u_bases=u_bases,
                    npts=npts + npts // 100,
                    num_reg_points=500,
                    l2_regularizer=l2_regularizer,
                    c=c,
                    xz_PML=xz_PML
                )
    else:
        if not use_PML:
            loss_train, loss_val, reg_loss, error_val = train_step(
                **common_args
            )
        if use_PML:
            loss_train, loss_val, reg_loss, error_val = train_step_PML(
                **common_args,
                omega0=omega0,
                c=c,
                xz_PML=xz_PML
            )

    Loss.append(loss_train)
    constraint_Loss.append(reg_loss)
    
    if cal_error:
        Error_val.append(error_val)
    if compute_validation:
        Loss_val.append(loss_val)
        
    # Check if loss_train is NaN, resulted from ill-posed LS
    if tf.math.is_nan(loss_train):
        print(f"Stopping training due to NaN loss at epoch {epoch}")
        break

    if epoch%100==0:
        print(" Epoch %d of %d" % (int(epoch), int(num_epochs)), end='\n')

        print(" Loss total: %.4f, Validation: %.4f, ConstraintLoss: %.4f, Val Error: %.4f,   Time taken: %.1fs" 
                  % (float(loss_train),float(loss_val),float(reg_loss),float(error_val),time.time()-epoch_time),end='\n')
        epoch_time=time.time()
    if (epoch % 5000 == 0) and saving:
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Convert elapsed time to minutes and seconds
        minutes, seconds = divmod(elapsed_time, 60)
        formatted_time = f'{int(minutes)} min {seconds:.0f} sec'
        save_model_and_history(epoch, formatted_time, u_model, Loss, Loss_val,Error_val,constraint_Loss,parameters)
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
# Convert elapsed time to minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)
formatted_time = f'{int(minutes)} min {seconds:.0f} sec'
print(f'Training time: {formatted_time}')

#% final saving<<<<<<<<<<<<<<<
save_model_and_history(epoch, formatted_time, u_model, Loss, Loss_val,Error_val,constraint_Loss,parameters)


#%%
#!!! %% Loading and Plotting
if plotting:
    model_path = "Results//Models/u_model_epoch_{epoch}.keras"
    u_model_loaded =keras.models.load_model(model_path,
        custom_objects={

            'EmbedderLayer': EmbedderLayer,
            'sin_activation': sin_activation}, compile=False)
    xz_in=xz_val
    u_model_loaded.summary()
    
    # Compute U0
    c_lims=[-np.max(np.abs(dU_2d)),np.max(np.abs(dU_2d))]
    U0_val = compute_U0(xz_val,s_xz, v0, omega,factor)
    plot_model_wavefield(dU_2d, xz_val,npts_x_val, npts_z_val,domain_bounds_valid)
    plt.clim(c_lims)
    plt.suptitle('FD modeled')

    u_predict = u_model_loaded(xz_in)
    plot_model_wavefield(u_predict, xz_val, npts_x_val, npts_z_val,domain_bounds_valid)
    plt.clim(c_lims)
    plt.suptitle('Prediction')
      
    # To load the history
    loaded_history = np.load('Results//training_history.npy', allow_pickle=True).item()
    
    Loss =loaded_history['training_loss']
    Loss_val =loaded_history['validation_loss']
    Error_val =loaded_history['validation_error']
    parameters =loaded_history['parameters']
    constraint_Loss=loaded_history['constraint_Loss']

        
    print(parameters)

        
    epoch_list_val=np.linspace(0,(len(Loss)-1),len(Loss_val))
    plt.figure(figsize=[6,4])
    plt.plot((Loss), 'k', label='Training')
    plt.plot(epoch_list_val,Loss_val, 'r:', label='Validation',alpha=1)
    plt.plot(constraint_Loss, 'g:', label='loss_forward',alpha=1)
    plt.xlim([num_epochs-100,num_epochs+.1])
    plt.yscale("log")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=[6,4])
    plt.plot(epoch_list_val,Error_val, 'b', label='Validation')
    plt.xlim([-1000,num_epochs+.1])
    plt.ylim([.0002,.025])
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.tight_layout()  
    plt.legend()
    plt.show()

