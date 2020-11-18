import math
import numpy as np
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow as tf

PINN = tfk.Sequential([
    tfk.layers.Input(2),
    tfk.layers.Dense(100, activation=tfk.activations.tanh),
    tfk.layers.Dense(100, activation=tfk.activations.tanh),
    tfk.layers.Dense(100, activation=tfk.activations.tanh),
    tfk.layers.Dense(100, activation=tfk.activations.tanh),
    tfk.layers.Dense(100, activation=tfk.activations.tanh),
    tfk.layers.Dense(2, activation=None),
    tfk.layers.Lambda(lambda x: tf.complex(x[..., 0], x[..., 1]))
])

sgd_opt = tfk.optimizers.SGD()

@tf.function
def train_colloc (t, x):
    """Trains the model to obey the given PDE at collocation points

    Args:
        t: A tf.Tensor of shape (batch_size,)
        x: A tf.Tensor of shape (batch_size,).
    """
    with tf.GradientTape(True, False) as tape:
        tape.watch(PINN.trainable_weights)
        #Calculate various derivatives of the output
        with tf.GradientTape(True, False) as grtape0:
            grtape0.watch([t, x])
            with tf.GradientTape(True, False) as grtape1:
                grtape1.watch([t, x])
                #Automatic differentiation of complex functions is weird in tensorflow
                #so we differentiate real and imaginary parts seperately
                h_real = tfm.real(PINN(tf.stack([t, x], -1)))
                h_imag = tfm.imag(PINN(tf.stack([t, x], -1)))
            #First order derivatives
            h_x_real = grtape1.gradient(h_real, x)
            h_x_imag = grtape1.gradient(h_imag, x)
            h_t_real = grtape1.gradient(h_real, t)
            h_t_imag = grtape1.gradient(h_imag, t)
            #h1_real and h1_imag have shape (batch_size,2)
            del grtape1
        #Second order derivatives
        h_xx_real = grtape0.gradient(h_x_real, x)
        h_xx_imag = grtape0.gradient(h_x_imag, x)
        del grtape0
        h = tf.complex(h_real, h_imag)
        h_t = tf.complex(h_t_real, h_t_imag)
        h_xx = tf.complex(h_xx_real, h_xx_imag)
        j = tf.complex(0,1)

        MSE = tfm.reduce_euclidean_norm(tfm.abs ((j*h_t) + (0.5*h_xx) + (tfm.conj(h)*h*h)))
    
    grads = tape.gradient (MSE, PINN.trainable_weights)
    sgd_opt.apply_gradients (zip(grads,PINN.trainable_weights))
    del tape
    return MSE

@tf.function
def train_init (t, x):
    """Trains the model to have a fixed initial condition when t=0

    Args:
        t: A tf.Tensor of shape (batch_size,)
        x: A tf.Tensor of shape (batch_size,).
    """
    def sech(x):
        return tf.complex (tfm.reciprocal(tfm.cosh(x)), 0)

    with tf.GradientTape() as tape:
        h = PINN (tf.stack ([t,x],-1))
        MSE = tfm.reduce_euclidean_norm (tfm.abs(h - sech(x)))
    grads = tape.gradient(MSE, PINN.trainable_weights)
    sgd_opt.apply_gradients(zip(grads, PINN.trainable_weights))
    return MSE

@tf.function
def train_bound (t):
    """Trains the model to equalize values and spatial derivatives at boundaries x=5 
    and x=-5 to enforce periodic boundary condition

    Args:
        t : A tf.Tensor of shape (batch_size,).
    """

    x1 = 5*tf.ones (t.shape)
    x2 = -5*tf.ones (t.shape)
    with tf.GradientTape(True, False) as tape:
        tape.watch (PINN.trainable_weights)
        with tf.GradientTape(True, False) as grtape1:
            grtape1.watch ([t, x1, x2])
            #Automatic differentiation of complex functions is weird in tensorflow
            #so we differentiate real and imaginary parts seperately
            h_real_1 = tfm.real(PINN(tf.stack([t, x1], -1)))
            h_imag_1 = tfm.imag(PINN(tf.stack([t, x1], -1)))
            h_real_2 = tfm.real(PINN(tf.stack([t, x2], -1)))
            h_imag_2 = tfm.imag(PINN(tf.stack([t, x2], -1)))
        #First order derivatives
        h_x1_real = grtape1.gradient(h_real_1, x1)
        h_x1_imag = grtape1.gradient(h_imag_1, x1)
        h_x2_real = grtape1.gradient(h_real_2, x2)
        h_x2_imag = grtape1.gradient(h_imag_2, x2)
        #h1_real and h1_imag have shape (batch_size,2)
        del grtape1
        h1 = tf.complex (h_real_1,h_imag_1)
        h1_x = tf.complex (h_x1_real, h_x1_imag)
        h2 = tf.complex(h_real_2, h_imag_2)
        h2_x = tf.complex(h_x2_real, h_x2_imag)
        MSE = tfm.reduce_mean(
                    tfm.pow(tfm.abs(h1-h2),2) + tfm.pow(tfm.abs(h1_x-h2_x),2))
    grads = tape.gradient(MSE, PINN.trainable_weights)
    sgd_opt.apply_gradients(zip(grads, PINN.trainable_weights))
    return MSE

def training_loop (epochs, batch_size = 10, Ni = 50, Nb = 50, Nf = 20000):
    """Randomly generates points on spacetime that PINN needs to train on
    runs the training loop for PINN.

    Args:
        epochs (int): Number of epochs in training loop.
        batch_size (int, optional): Size of training batch. Defaults to 25.
        Ni (int, optional): Number of initial data. Defaults to 50.
        Nb (int, optional): Number of boundary data. Defaults to 50.
        Nf (int, optional): Number of collocation data. Defaults to 20000.
    """
    Ni_b = Ni // batch_size
    Nb_b = Nb // batch_size
    Nf_b = Nf // batch_size
    data_len = Ni_b + Nb_b + Nf_b
    pos = np.random.choice (data_len, Ni_b + Nb_b, False)
    init_pos = pos[:Ni_b]
    bound_pos = pos[Ni_b:]

    for _ in range (epochs):
        for step in range (data_len):
            if step in init_pos:
                t = tf.zeros (batch_size)
                x = tf.random.uniform ((batch_size,), -5, 5)
                loss = train_init(t, x)
            elif step in bound_pos:
                t = tf.random.uniform ((batch_size,), 0, math.pi/2)
                loss = train_bound (t)
            else:
                t = tf.random.uniform((batch_size,), 0, math.pi/2)
                x = tf.random.uniform((batch_size,), -5, 5)
                loss = train_colloc (t, x)
            if step % 50 == 0:
                print ()
