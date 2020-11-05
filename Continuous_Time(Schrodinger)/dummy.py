import numpy as np
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow as tf

#There's no hyperbolic secant op in Tensorflow so we're defining a custom one
def my_numpy_func(x):
    return np.reciprocal(np.cosh(x))

@tf.function
def sech(x):
    y = tf.numpy_function(my_numpy_func, [x], tf.complex)
    return y

@tf.function
def tf_diag (v, k=0):
    y = tf.numpy_function(np.diag, [v,k], tf.float32)
    return y

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
def train_colloc (input_var):
    with tf.GradientTape(True, False) as tape:
        tape.watch (PINN.trainable_weights)
        #Calculate various derivatives of the output
        with tf.GradientTape(True, False) as grtape0:
            grtape0.watch(input_var)
            with tf.GradientTape(True, False) as grtape1:
                grtape1.watch(input_var)
                #Automatic differentiation of complex functions is weird in tensorflow
                #so we differentiate real and imaginary parts seperately
                h_real = tfm.real(PINN(input_var))
                h_imag = tfm.imag(PINN(input_var))
            #First order derivatives
            h1_real = grtape1.gradient(h_real, input_var)
            h1_imag = grtape1.gradient(h_imag, input_var)
            #h1_real and h1_imag have shape (batch_size,2)
            del grtape1
        #Second order derivatives
        h2_real = grtape0.jacobian(h1_real, input_var)
        h2_imag = grtape0.jacobian(h1_real, input_var)
        #h2_real and h2_imag have shape (batch_size,2,batch_size,2)
        del grtape0
        h = tf.complex(h_real, h_imag)
        h_t = tf.complex(h1_real[:, 0], h1_imag[:, 0])
        h_xx = tf_diag(tf.complex(h2_real[:, 1, :, 1], h2_imag[:, 1, :, 1]))
        j = tf.complex(0,1)

        MSE = tfm.reduce_euclidean_norm( tfm.abs ((j*h_t) + (0.5*h_xx) + (tfm.conj(h)*h*h)) )

    x = input_var[:, 1]
    t = input_var[:, 0]
    grads = tape.gradient (MSE, PINN.trainable_weights)
    sgd_opt.apply_gradients (zip(grads,PINN.trainable_weights))
    del tape
