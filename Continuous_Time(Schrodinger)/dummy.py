import math
import numpy as np
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow as tf

#There's no hyperbolic secant op in Tensorflow so we're defining a custom one
def my_numpy_func(x):
    return np.reciprocal(np.cosh(x))

@tf.function
def sech(x):
    tfm.reciprocal (tfm.cosh (x))
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
    """Trains the model to obey the given PDE at collocation points

    Args:
        input_var: A tf.Tensor of shape (batch_size,2). Components of second index
                    denote x and t
    """
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

        MSE = tfm.reduce_euclidean_norm(tfm.abs ((j*h_t) + (0.5*h_xx) + (tfm.conj(h)*h*h)))
    
    grads = tape.gradient (MSE, PINN.trainable_weights)
    sgd_opt.apply_gradients (zip(grads,PINN.trainable_weights))
    del tape

@tf.function
def train_init (input_var):
    """Trains the model to have a fixed initial condition when t=0

    Args:
        input_var : A tf.Tensor of shape (batch_size,2). Components of second index
                    denote x and t. Value of t should always be 0.
    """
    with tf.GradientTape() as tape:
        h = PINN (input_var)
        MSE = tfm.reduce_euclidean_norm ( tfm.abs(h - sech(input_var[:,1])) )
    grads = tape.gradient(MSE, PINN.trainable_weights)
    sgd_opt.apply_gradients(zip(grads, PINN.trainable_weights))

@tf.function
def train_bound (input_times):
    """Trains the model to equalize values and spatial derivatives at boundaries x=5 
    and x=-5 to enforce periodic boundary condition

    Args:
        input_times : A tf.Tensor of shape (batch_size,). Denotes values of t
    """

    input_var1 = tf.stack([input_times, 5*tf.ones(input_times.shape)], axis=-1)
    input_var2 = tf.stack([input_times,-5*tf.ones(input_times.shape)], axis=-1)
    with tf.GradientTape(True, False) as tape:
        tape.watch (PINN.trainable_weights)
        with tf.GradientTape(True, False) as grtape1:
            grtape1.watch ([input_var1, input_var2])
            #Automatic differentiation of complex functions is weird in tensorflow
            #so we differentiate real and imaginary parts seperately
            h_real_1 = tfm.real(PINN(input_var1))
            h_imag_1 = tfm.imag(PINN(input_var1))
            h_real_2 = tfm.real(PINN(input_var2))
            h_imag_2 = tfm.imag(PINN(input_var2))
        #First order derivatives
        h1_real_1 = grtape1.gradient(h_real_1, input_var1)
        h1_imag_1 = grtape1.gradient(h_imag_1, input_var1)
        h1_real_2 = grtape1.gradient(h_real_2, input_var2)
        h1_imag_2 = grtape1.gradient(h_imag_2, input_var2)
        #h1_real and h1_imag have shape (batch_size,2)
        del grtape1
        h1 = tf.complex (h_real_1,h_imag_1)
        h1_x = tf.complex (h1_real_1[:,1], h1_imag_1[:,1])
        h2 = tf.complex(h_real_2, h_imag_2)
        h2_x = tf.complex(h1_real_2[:, 1], h1_imag_2[:, 1])
        MSE = tfm.reduce_mean(
                    tfm.pow(tfm.abs(h1-h2),2) + tfm.pow(tfm.abs(h1_x-h2_x),2))
    grads = tape.gradient(MSE, PINN.trainable_weights)
    sgd_opt.apply_gradients(zip(grads, PINN.trainable_weights))

def training_loop (epochs, batch_size = 25, Ni = 50, Nb = 50, Nf = 20000):
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
    pos = np.random.choice (data_len, 4, False)
    init_pos = pos[:2]
    bound_pos = pos[2:]

    for _ in range (epochs):
        for step in range (data_len):
            if step in init_pos:
                in_times = tf.zeros (batch_size)
                in_pos = tf.random.uniform ((batch_size,), -5, 5)
                train_init (tf.stack ([in_times, in_pos], -1))
            elif step in bound_pos:
                in_times = tf.random.uniform ((batch_size,), 0, math.pi/2)
                train_bound (in_times)
            else:
                in_pos = tf.random.uniform((batch_size,), -5, 5)
                in_times = tf.random.uniform((batch_size,), 0, math.pi/2)
                train_colloc (tf.stack([in_times, in_pos], -1))
