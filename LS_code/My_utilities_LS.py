#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:13:09 2025

@author: mabedi
"""
import tensorflow as tf
import numpy as np
from scipy.special import hankel1,hankel2  # Import Bessel and Hankel functions of the first and second kind (order 0)
import matplotlib.pyplot as plt
import scipy.interpolate

# Define the background wavefield U0 in 2D
def compute_U0(xz, s_xz, v0, omega,factor=1.):
    """
    Compute the background wavefield U0 for the 2D Helmholtz equation.
    
    Args:
    x, z: Tensors of spatial coordinates (same shape).
    sx, sz: Source location (scalars).
    v0: Constant background velocity.
    omega: Angular frequency.
    factor: obtained by matching the analytical and finite-difference magnitudes
    
    Returns:
    U0: The background wavefield.
    """
    x, z = tf.unstack(xz, axis=-1)  # x and z will have shape [batch_size]
    x = tf.reshape(x, (-1, 1))  # Shape [batch_size, 1]
    z = tf.reshape(z, (-1, 1))  # Shape [batch_size, 1]
    sx, sz = tf.unstack(s_xz, axis=-1)
    # Compute the distance between the point and the source
    r = tf.sqrt((x - sx)**2 + (z - sz)**2)
    # print('r:',np.shape(r))
    # Compute the argument for the Hankel function
    # arg = omega * r / v0
    # Avoid division by zero by assigning a specific value when r is zero
    arg = tf.where(r == 0, tf.constant(1e-9, dtype=r.dtype), omega * r / v0)

    # Compute the background wavefield U0
    # U0 = (1j / 4) * hankel_0_second_kind(arg)
    U0 = factor*(1j / 4) *hankel2(0,arg)
    U0_real = tf.math.real(U0)  # Shape: [batch_size, 1]
    U0_imag = tf.math.imag(U0)  # Shape: [batch_size, 1]
    # print('U0:',np.shape(U0))
    U0=tf.concat([U0_real,U0_imag],axis=-1)
    # print('U0 stack real and imaginary:',np.shape(U0))
    return U0

#2D interpolation of the velocity modl for given collocation points:
def interpolator(v,domain_bounds,xz,dtype=tf.float32):
    a_x,b_x,a_z,b_z=domain_bounds
    nz,nx=np.shape(v)
    # Create a grid of the original coordinates for v
    x_orig = np.linspace(a_x, b_x, nx)
    z_orig = np.linspace(a_z, b_z, nz)
    X_orig, Z_orig = np.meshgrid(x_orig, z_orig)
    
    # Flatten the grid and v
    points_orig = np.column_stack([X_orig.ravel(), Z_orig.ravel()])
    v_flat = v.ravel()
    
    # Interpolate using scipy's griddata
    v_interpolated = scipy.interpolate.griddata(points_orig, v_flat, xz, method='linear')

    # Convert to TensorFlow tensor (optional)
    v_interpolated = tf.convert_to_tensor(v_interpolated, dtype=dtype)
    v_interpolated = tf.reshape(v_interpolated, (-1, 1))
    return v_interpolated

def create_xz_reg(xz, s_xz, omega, v0,beta=1000, num_reg_points=500):
    """
    Select the closest 500 points in `xz` to `s_xz` for near source regularization.

    Args:
    xz: Tensor of coordinates (shape: [num_points, 2]).
    s_xz: Source location coordinates (shape: [1, 2]).
    omega: Angular frequency.
    v0: Background velocity.
    num_reg_points: Number of closest points to select for regularization (default: 500).

    Returns:
    xz_reg: The 500 closest points to `s_xz` (Tensor).
    factor_d: Regularization factors for the selected points (Tensor).
    """
    # Compute squared distances from each point in xz to the source location s_xz
    distance_squared =( tf.reduce_sum(tf.square(xz - s_xz), axis=-1))
    
    # Get the indices of the closest 500 points
    _, indices = tf.nn.top_k(-distance_squared, k=num_reg_points)  # Use negative to get smallest distances
    
    # Gather the closest points based on the indices
    xz_reg = tf.gather(xz, indices)

    # Define the maximum distance (squared) for regularization (lambda/2)
    max_distance_squared = tf.cast(( (v0 * 0.5 / (omega/(2*3.141592))) ** 2),dtype='float32')# 0.25 for the constrint region lambda/4 

    # Calculate the regularization factor for the selected points
    # factor_d = tf.expand_dims(tf.gather(tf.nn.relu(max_distance_squared - distance_squared) * beta**2 * omega, indices),axis=-1)#1e6
    #frequency dependent soft-constraint coefficient = f^2*beta: 
    factor_d = tf.expand_dims(tf.gather(tf.nn.relu(max_distance_squared - distance_squared) * (beta *(omega/(2*3.141592))**2)**2, indices),axis=-1)#1e6

    return xz_reg, factor_d,indices


def save_model_and_history(epoch, formatted_time, u_model, Loss, Loss_val, Error_val,constratint_Loss, parameters):
    print('saving...')
    # Update only the training time
    parameters["Training_time"] = formatted_time

    # Dynamically include the epoch number in the model filename
    model_filename = f'Results/Models/u_model_epoch_{epoch}.keras'

    # Save the model and training history
    u_model.save(model_filename)
    history = {
        "training_loss": Loss,
        "validation_loss": Loss_val,
        "validation_error": Error_val,
        "constratint_Loss":constratint_Loss,
        "parameters": parameters
    }
    np.save('Results/training_history.npy', history)
    print('\rSaved!       ')
    

def sin_activation(x):
    return tf.sin(x)

#Plot real and imaginary parts of the wavefield
def plot_model_wavefield(wavefield, xz, npts_x, npts_z,domain_bounds):
    a_x,b_x,a_z,b_z=domain_bounds
    # Extract the real and imaginary parts
    u_real = wavefield[:, 0]  # Real part
    u_imag = wavefield[:, 1]  # Imaginary part

    # Reshape the real part wavefield back into a 2D grid
    u_real_grid = tf.reshape(u_real, (npts_z, npts_x))  # Shape [npts_z, npts_x]
    
    # Reshape the imaginary part wavefield back into a 2D grid
    u_imag_grid = tf.reshape(u_imag, (npts_z, npts_x))  # Shape [npts_z, npts_x]

    # Plot the real part as a 2D image
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.imshow(u_real_grid, extent=[a_x, b_x, b_z, a_z], origin='upper', cmap='viridis', aspect='auto')
    plt.colorbar(label='Real Part')
    plt.title("Real Part")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()

    # Plot the imaginary part as a 2D image
    plt.subplot(122)
    plt.imshow(u_imag_grid, extent=[a_x, b_x,b_z, a_z], origin='upper', cmap='plasma', aspect='auto')
    plt.colorbar(label='Imaginary Part')
    plt.title("Imaginary Part")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
    # plt.tight_layout()
    
#!!!
# Define the loss function for the 2D Helmholtz equation (separating real and imaginary parts)
@tf.function()
def make_loss(u_model, U0, xz, v, v0, omega, lamda,use_source_reg,source_reg,cal_error=False,dU_2d_val=0.,use_Vin=False,num_reg_points=500):
    # # tf.print("xz",tf.shape(xz))
    x, z = tf.split(xz, num_or_size_splits=2, axis=-1)  # x and z each have shape [batch_size, 1]
    # tf.print("x",tf.shape(x))
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, z])  # Watch both x and z
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, z])
            xz = tf.concat([x, z], axis=-1)  # Shape: [npts, 2]

            u = u_model(xz)  # Scattered wavefield (delta U), 2D outputs: real and imaginary

            # Split u into real and imaginary parts
            u_real = u[:, 0:1]  # Real part
            u_imag = u[:, 1:2]  # Imaginary part

        # Compute the first derivatives w.r.t both x and z for real and imaginary parts
        u_x_real,u_z_real = tape2.gradient(u_real, [x,z])  # First derivative w.r.t x (shape: [batch_size, 1])
        u_x_imag,u_z_imag = tape2.gradient(u_imag, [x,z])  # First derivative w.r.t x (imaginary)

    # Compute the second derivatives (Laplacian components) for real and imaginary parts
    u_xx_real = tape1.gradient(u_x_real, x)  # Second derivative w.r.t x
    u_zz_real = tape1.gradient(u_z_real, z)  # Second derivative w.r.t z
    u_xx_imag = tape1.gradient(u_x_imag, x)  # Second derivative w.r.t x (imaginary)
    u_zz_imag = tape1.gradient(u_z_imag, z)  # Second derivative w.r.t z (imaginary)

    # Compute the 2D Laplacian (sum of second derivatives w.r.t x and z) for both real and imaginary parts
    laplacian_u_real = u_xx_real + u_zz_real  # Real part of Laplacian
    laplacian_u_imag = u_xx_imag + u_zz_imag  # Imaginary part of Laplacian
    # Clean up tapes
    del tape1, tape2

    # Split real and imaginary parts of U0 
    U0_real = U0[:,0:1]  # Shape: [batch_size, 1]
    U0_imag = U0[:,1:2]  # Shape: [batch_size, 1]

    # Helmholtz equation residual for real and imaginary parts in 2D
    helmholtz_residual_real = omega**2 * (1 / v**2) * u_real + laplacian_u_real + omega**2 * (1 / v**2 - 1 / v0**2) * U0_real
    helmholtz_residual_imag = omega**2 * (1 / v**2) * u_imag + laplacian_u_imag + omega**2 * (1 / v**2 - 1 / v0**2) * U0_imag
    # tf.print('helmholtz_residual_real:', np.shape(helmholtz_residual_real))
    # Loss term from the Helmholtz equation residual for real and imaginary parts
    pde_loss_real = lamda * tf.reduce_mean(tf.square(helmholtz_residual_real))
    pde_loss_imag = lamda * tf.reduce_mean(tf.square(helmholtz_residual_imag))

    # Total loss is the sum of both real and imaginary losses
    total_loss = pde_loss_real + pde_loss_imag
    
    if use_source_reg:# Regularization loss term: Penalize the scattered wavefield near the source, scaled by `factor_d`

        # Compute scattered wavefield at regularization points
        if use_Vin:
            xz_reg, factor_d,v_reg=source_reg
            u_reg = u_model  ([xz_reg,v_reg]) 
        else:
            xz_reg, factor_d,_=source_reg
            u_reg = u_model  (xz_reg)

        reg_loss = 1/num_reg_points*tf.reduce_sum(tf.square(tf.sqrt(factor_d )* u_reg))
        total_loss = total_loss+reg_loss
    else:
        reg_loss=0
    if cal_error:    
        error=tf.reduce_mean(tf.abs(dU_2d_val-u))
        # error=tf.reduce_mean(tf.square(dU_2d_val-u))
    else:
        error=0.
    return total_loss,reg_loss,error

# PML %%%%%%%%%%%%%%%%%%%%%%%

@tf.function
def make_loss_PML(u_model, U0, xz, v, v0, omega, lamda, omega0, c,xz_PML, use_source_reg, source_reg, cal_error=False, dU_2d_val=0.0):

    xb1,xb2,zb1,zb2=xz_PML#boundaries coordinates
    x, z = tf.split(xz, num_or_size_splits=2, axis=-1)  # Split x and z coordinates
    
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, z])
        lx = tf.nn.relu(xb1 - x) +tf.nn.relu(x - xb2)  # Distance to PML boundary 
        lz = tf.nn.relu(zb1 - z) +tf.nn.relu(z - zb2 )
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, z])
            xz = tf.concat([x, z], axis=-1)  # Shape: [npts, 2]
            u = u_model(xz)  # Model predictions: delta u (real and imaginary)
            u_real, u_imag = tf.split(u, num_or_size_splits=2, axis=-1)

        # First derivatives
        u_x_real, u_z_real = tape2.gradient(u_real, [x, z])
        u_x_imag, u_z_imag = tape2.gradient(u_imag, [x, z])
        
    # Compute the second derivatives (Laplacian components) for real and imaginary parts
    u_xx_real = tape1.gradient(u_x_real, x)  # Second derivative w.r.t x
    u_zz_real = tape1.gradient(u_z_real, z)  # Second derivative w.r.t z
    u_xx_imag = tape1.gradient(u_x_imag, x)  # Second derivative w.r.t x (imaginary)
    u_zz_imag = tape1.gradient(u_z_imag, z)  # Second derivative w.r.t z (imaginary)
    lx_x=tape1.gradient(lx, x)  
    lz_z=tape1.gradient(lz, z) 
    
    
    # Definitions of parameters
    Cxr = (1 + c**2 * lx**2 * lz**2) / (1 + c**2 * lx**4)
    Cxi = (c * (lx**2 - lz**2)) / (1 + c**2 * lx**4)
    Czr = (1 + c**2 * lx**2 * lz**2) / (1 + c**2 * lz**4)
    Czi = (c * (-lx**2 + lz**2)) / (1 + c**2 * lz**4)

    # Definitions of derivatives of above parameters
    dCxr = (2 * c**2 * lx * (-2 * lx**2 + lz**2 - c**2 * lx**4 * lz**2) * lx_x) / (1 + c**2 * lx**4)**2
    dCxi = (2 * c * lx * (1 - c**2 * lx**2 * (lx**2 - 2 * lz**2)) * lx_x) / (1 + c**2 * lx**4)**2
    dCzr = -(2 * c**2 * (2 * lz**3 + lx**2 * lz * (-1 + c**2 * lz**4)) * lz_z) / (1 + c**2 * lz**4)**2
    dCzi = (2 * c * lz * (1 - c**2 * lz**2 * (-2 * lx**2 + lz**2)) * lz_z) / (1 + c**2 * lz**4)**2

    Fr_xx=dCxr*u_x_real + Cxr*u_xx_real - dCxi*u_x_imag - Cxi*u_xx_imag
    Fr_zz=dCzr*u_z_real + Czr*u_zz_real - dCzi*u_z_imag - Czi*u_zz_imag
    Fi_xx=dCxr*u_x_imag + Cxr*u_xx_imag + dCxi*u_x_real + Cxi*u_xx_real
    Fi_zz=dCzr*u_z_imag + Czr*u_zz_imag + dCzi*u_z_real + Czi*u_zz_real

    del tape1, tape2

    # Extract U0 components
    U0_real, U0_imag = tf.split(U0, num_or_size_splits=2, axis=-1)
    # non_PML_region= 1-tf.math.sign(lx+lz) # Distance to PML boundary 
    U0_decay_factor = tf.math.exp(-omega/v0 * c * (tf.sqrt(lx**2+lz**2)**3) / 3)
    # Define auxiliary terms for Fr and Fi
    omega2 = omega**2
    m=v**-2
    m0=v0**-2
    term1_real = (1 - c**2 * lx**2 * lz**2) * omega2 * (m * u_real + (m - m0) * U0_real*U0_decay_factor)
    term2_real = c * (lx**2 + lz**2)        * omega2 * (m * u_imag + (m - m0) * U0_imag*U0_decay_factor)
    term1_imag = (1 - c**2 * lx**2 * lz**2) * omega2 * (m * u_imag + (m - m0) * U0_imag*U0_decay_factor)
    term2_imag = -c * (lx**2 + lz**2)       * omega2 * (m * u_real + (m - m0) * U0_real*U0_decay_factor)


    # Fr and Fi terms
    Fr = (Fr_xx + Fr_zz + term1_real + term2_real)
    Fi = (Fi_xx + Fi_zz + term1_imag + term2_imag)

    # Compute losses
    loss_Fr = tf.reduce_mean(tf.square(Fr))
    loss_Fi = tf.reduce_mean(tf.square(Fi))

    # Total loss
    total_loss = lamda * (loss_Fr + loss_Fi)

    if use_source_reg:# Regularization loss term (soft constraint): Penalize the scattered wavefield near the source, scaled by `factor_d`

        # Compute scattered wavefield at regularization points
        xz_reg, factor_d,_=source_reg
        u_reg = u_model(xz_reg)
    
        # Regularization loss (only consider valid points using the mask)
        reg_loss = 2*tf.reduce_mean( factor_d * tf.square(u_reg))
        total_loss = total_loss+reg_loss
    else:
        reg_loss=0
    if cal_error:    
        error=tf.reduce_mean(tf.abs(dU_2d_val-u))
        # error=tf.reduce_mean(tf.square(dU_2d_val-u))
    else:
        error=0.
    return total_loss,reg_loss,error


@tf.function
def LS_diff_PML(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,model_type,c,xz_PML,use_source_reg,source_reg,num_reg_points,use_Vin):
    # Split input coordinates into x and z components (2D input: [batch_size, 2])
    xb1,xb2,zb1,zb2=xz_PML#boundaries
    x, z = tf.split(xz, num_or_size_splits=2, axis=-1)  # Split x and z coordinates

    # Forward mode autodiff to compute the derivatives with respect to x and z
    with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t1_x, tf.autodiff.ForwardAccumulator(z, tf.ones_like(z)) as t1_z:
        lx = tf.nn.relu(xb1 - x) +tf.nn.relu(x - xb2)  # Distance to PML boundary 
        lz = tf.nn.relu(zb1 - z) +tf.nn.relu(z - zb2 )
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t2_x, tf.autodiff.ForwardAccumulator(z, tf.ones_like(z)) as t2_z:
            xz = tf.concat([x, z], axis=-1)  # Shape: [npts, 2]
            u_b = u_bases(xz)  # penultimate layer output

        # First derivatives w.r.t x and z
        u_x_b = t2_x.jvp(u_b)   # First derivative w.r.t x
        u_z_b = t2_z.jvp(u_b)   # First derivative w.r.t z
    # Second derivatives w.r.t x and z
    u_xx_b = t1_x.jvp(u_x_b)  # Second derivative w.r.t x
    u_zz_b = t1_z.jvp(u_z_b)  # Second derivative w.r.t z
    lx_x=t1_x.jvp(lx)  
    lz_z=t1_z.jvp(lz) 

    # Definitions of common terms
    Cxr = (1 + c**2 * lx**2 * lz**2) / (1 + c**2 * lx**4)
    Cxi = (c * (lx**2 - lz**2)) / (1 + c**2 * lx**4)
    Czr = (1 + c**2 * lx**2 * lz**2) / (1 + c**2 * lz**4)
    Czi = (c * (-lx**2 + lz**2)) / (1 + c**2 * lz**4)

    # Definitions of derivatives of above parameters
    dCxr = (2 * c**2 * lx * (-2 * lx**2 + lz**2 - c**2 * lx**4 * lz**2) * lx_x) / (1 + c**2 * lx**4)**2
    dCxi = (2 * c * lx * (1 - c**2 * lx**2 * (lx**2 - 2 * lz**2)) * lx_x) / (1 + c**2 * lx**4)**2
    dCzr = -(2 * c**2 * (2 * lz**3 + lx**2 * lz * (-1 + c**2 * lz**4)) * lz_z) / (1 + c**2 * lz**4)**2
    dCzi = (2 * c * lz * (1 - c**2 * lz**2 * (-2 * lx**2 + lz**2)) * lz_z) / (1 + c**2 * lz**4)**2

    Fr_x_r=dCxr*u_x_b + Cxr*u_xx_b 
    Fr_x_i=- dCxi*u_x_b - Cxi*u_xx_b
    Fr_z_r=dCzr*u_z_b + Czr*u_zz_b 
    Fr_z_i=- dCzi*u_z_b - Czi*u_zz_b
    
    # tf.print('Fr_x_r',np.shape(Fr_x_r))
    U0_real, U0_imag = tf.split(U0, num_or_size_splits=2, axis=-1)
    U0_decay_factor = tf.math.exp(-omega/v0 * c * (tf.sqrt(lx**2+lz**2)**3) / 3)
    # Define auxiliary terms for Fr and Fi
    omega2 = omega**2
    m=v**-2
    m0=v0**-2
    
    term_r_r = (1 - c**2 * lx**2 * lz**2) * omega2 * (m * u_b )
    term_r_i = c * (lx**2 + lz**2)       * omega2 * (m * u_b )

    R_r_r = (1 - c**2 * lx**2 * lz**2) * omega2 * ((m - m0) * U0_real*U0_decay_factor)
    R_r_i = c * (lx**2 + lz**2)       * omega2 * ((m - m0) * U0_imag*U0_decay_factor)
    R_i_i = (1 - c**2 * lx**2 * lz**2) * omega2 * ((m - m0) * U0_imag*U0_decay_factor)
    R_i_r = -c * (lx**2 + lz**2)        * omega2 * ( (m - m0) * U0_real*U0_decay_factor)
 
    
    B1=Fr_x_r+Fr_z_r+term_r_r
    B2=Fr_x_i+Fr_z_i+term_r_i
    B3=-B2
    B4=B1
    
    R_r=-(R_r_r+R_r_i)
    R_i=-(R_i_i+R_i_r)

    
    # Concatenate horizontally
    B =tf.sqrt(1/npts)*tf.concat([
        tf.concat([B1, B3], axis=0),  # Stack B1 over B3 
        tf.concat([B2, B4], axis=0)   # Stack B2 over B4 
    ], axis=1)  # Concatenate them horizontally 
    R=tf.sqrt(1/npts)*tf.concat([R_r, R_i], axis=0)

    if use_source_reg:

        xz_reg, factor_d,_=source_reg
        u_b_reg = u_bases(xz_reg) 

        u_b_reg = tf.sqrt(1/num_reg_points)*tf.sqrt(factor_d)*u_b_reg
        u_b_reg=tf.concat([u_b_reg, u_b_reg], axis=1)
        B = tf.concat([B,u_b_reg], axis=0)#batch is axis=0
        R = tf.concat([R,tf.zeros([num_reg_points,1])], axis=0)

    return B,R

#!!!
@tf.function
def LS_diff(u_model, u_bases, U0, xz, v, v0, omega, lamda,npts,model_type,use_source_reg=False,source_reg=0,num_reg_points=500,use_Vin=False):

    # Split input coordinates into x and z components (2D input: [batch_size, 2])
    x, z = tf.split(xz, num_or_size_splits=2, axis=-1)  # x and z each have shape [batch_size, 1]
    # Forward mode autodiff to compute the derivatives with respect to x and z
    with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t1_x, tf.autodiff.ForwardAccumulator(z, tf.ones_like(z)) as t1_z:
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t2_x, tf.autodiff.ForwardAccumulator(z, tf.ones_like(z)) as t2_z:
            xz = tf.concat([x, z], axis=-1)  # Shape: [npts, 2]

            u_b = u_bases(xz)  # penultimate layer ouput

        # First derivatives w.r.t x and z
        u_x_b = t2_x.jvp(u_b)   # First derivative w.r.t x
        u_z_b = t2_z.jvp(u_b)   # First derivative w.r.t z
    # Second derivatives w.r.t x and z
    u_xx_b = t1_x.jvp(u_x_b)  # Second derivative w.r.t x
    u_zz_b = t1_z.jvp(u_z_b)  # Second derivative w.r.t z
    # Laplacian in 2D (sum of second derivatives w.r.t x and z)
    laplacian_u_b = u_xx_b + u_zz_b  # Same shape as u_b

    B = tf.sqrt(1/npts)*(omega**2 / v**2 * u_b + laplacian_u_b)  # B shape: [batch_size, neurons_final]
    R = tf.sqrt(1/npts)*(-omega**2 * (1 / v**2 - 1 / v0**2) * U0)  # R shape: [batch_size, 2]

    if use_source_reg:

        if use_Vin:
            xz_reg, factor_d,v_reg=source_reg
            u_b_reg = u_bases([xz_reg,v_reg]) 
        else:
            xz_reg, factor_d,_=source_reg
            u_b_reg = u_bases(xz_reg) 
            

        u_b_reg = tf.sqrt(1/num_reg_points)*tf.sqrt(factor_d)*u_b_reg

        B = tf.concat([B,u_b_reg], axis=0)#batch is axis=0
        R = tf.concat([R,tf.zeros([num_reg_points,2])], axis=0)

    return B,R