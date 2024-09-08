import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers

np.set_printoptions(precision=2, suppress=True, linewidth=100, formatter={'complexfloat': '{:+.2f}'.format})


def normalize_data(data):
    data_min = np.min(data, axis=(1, 2, 3), keepdims=True)
    data_max = np.max(data, axis=(1, 2, 3), keepdims=True)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data


Tx = 8
Rx = 7
Vx = 2

print("The sample number is 10000")

for Dr in np.arange(0.1, 0.6, 0.1):
    print("The Dr is", Dr)

    file_path = f"/Sample(7+2_8)_Dr={Dr:.1f}_n=0.01_10e5.h5"

    # Load and pre-process the samples
    sample_number = 10000
    with h5py.File(file_path, 'r') as f:
        H_matrices = f['H'][:sample_number]
        Rx_fix_matrices = f['Rx_fix'][:sample_number]
        Rx_vir_matrices = f['Rx_vir'][:sample_number]

    # transform the complex matrix into double-channel matrix
    H_matrices_real_imag = np.stack((H_matrices.real, H_matrices.imag), axis=-1)
    Rx_fix_matrices_real_imag = np.stack((Rx_fix_matrices.real, Rx_fix_matrices.imag), axis=-1)
    Rx_vir_matrices_real_imag = np.stack((Rx_vir_matrices.real, Rx_vir_matrices.imag), axis=-1)

    # Padding Rx_fix_matrices_real with zeros
    # combine the padded Rx and H_matries_real_imag，get H_input, 为9*9*2
    Rx_add_zeros = np.zeros((sample_number, Rx + Vx, 1, 2))
    Rx_add_zeros[:, :Rx, :, :] = Rx_fix_matrices_real_imag
    H_input = np.concatenate((Rx_add_zeros, H_matrices_real_imag), axis=2)

    # Normalize all the double-channel inputs
    H_input = normalize_data(H_input)

    # define the input layer
    L2_parameter = 0.0001

    model = tf.keras.Sequential([tf.keras.layers.InputLayer(shape=(9, 9, 2)),

                                 # Convolutional layer
                                 # The kernel number decides the number of output channels
                                 tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same',
                                                        kernel_regularizer=regularizers.l2(L2_parameter)),
                                 tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same',
                                                        kernel_regularizer=regularizers.l2(L2_parameter)),
                                 tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same',
                                                        kernel_regularizer=regularizers.l2(L2_parameter)),

                                 # Flatten layer
                                 tf.keras.layers.Flatten(),

                                 # MLP
                                 tf.keras.layers.Dense(128, activation='tanh',
                                                       kernel_regularizer=regularizers.l2(L2_parameter)),
                                 tf.keras.layers.Dense(256, activation='tanh',
                                                       kernel_regularizer=regularizers.l2(L2_parameter)),
                                 tf.keras.layers.Dense(128, activation='tanh',
                                                       kernel_regularizer=regularizers.l2(L2_parameter)),

                                 # Output layer
                                 tf.keras.layers.Dense(2 * Vx, activation='linear'),
                                 tf.keras.layers.Reshape((Vx, 1, 2))
                                 ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse',
                  metrics=['mean_absolute_percentage_error'])

    # Train the model
    model.fit(H_input, Rx_vir_matrices_real_imag, epochs=10, batch_size=32, validation_split=0.2)

    # Save model
    # model.save('model_Rx=7+2_Tx=8_Dr=0.1_n=0.01_2.h5')
