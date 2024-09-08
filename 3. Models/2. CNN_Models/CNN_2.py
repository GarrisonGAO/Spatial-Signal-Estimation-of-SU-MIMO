import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras import layers

np.set_printoptions(precision=2, suppress=True, linewidth=100, formatter={'complexfloat': '{:+.2f}'.format})


# Normalization function, [0, 1]
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

    # Load and pre-process the data
    sample_number = 10000
    with h5py.File(file_path, 'r') as f:
        H_matrices = f['H'][:sample_number]
        Rx_fix_matrices = f['Rx_fix'][:sample_number]
        Rx_vir_matrices = f['Rx_vir'][:sample_number]

    # Transform from complex matrix to the double channel matrix
    H_matrices_real_imag = np.stack((H_matrices.real, H_matrices.imag), axis=-1)
    Rx_fix_matrices_real_imag = np.stack((Rx_fix_matrices.real, Rx_fix_matrices.imag), axis=-1)
    Rx_vir_matrices_real_imag = np.stack((Rx_vir_matrices.real, Rx_vir_matrices.imag), axis=-1)

    # Padding the Rx_fix_matrices_real with zeros
    # Combine the padded Rx and H_matries_real_imag，get H_input
    Rx_add_zeros = np.zeros((sample_number, Rx + Vx, 1, 2))
    Rx_add_zeros[:, :Rx, :, :] = Rx_fix_matrices_real_imag
    H_input = np.concatenate((Rx_add_zeros, H_matrices_real_imag), axis=2)

    # Normalize all the double-channel inputs
    H_input = normalize_data(H_input)
    Rx_fix_matrices_real_imag = normalize_data(Rx_fix_matrices_real_imag)

    # Define the input layers
    # first layer receive the combined channel matrix
    # second layer receive the Rx vector

    L2_parameter = 0.0001
    input_matrix_1 = tf.keras.layers.Input(shape=(Rx + Vx, Tx + 1, 2), name='input_image_1')
    input_matrix_2 = tf.keras.layers.Input(shape=(Rx, 1, 2), name='input_image_2')

    # Take the convolution operation over the channel matrix H
    x1 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(
        input_matrix_1)
    x1 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x1)
    x1 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    # Take the convolution operation over the Rx_fix
    x2 = tf.keras.layers.Conv2D(32, (3, 1), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(L2_parameter))(input_matrix_2)
    x2 = tf.keras.layers.Conv2D(64, (3, 1), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(L2_parameter))(x2)
    x2 = tf.keras.layers.Conv2D(128, (3, 1), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(L2_parameter))(x2)
    x2 = tf.keras.layers.Flatten()(x2)

    # combine the 2 ouputs
    merged = tf.keras.layers.concatenate([x1, x2])
    print(merged.shape)

    # MLP layers
    fc1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(merged)
    fc2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(fc1)
    fc3 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(fc2)

    # Output layer, get adapted with Rx_vir
    output = tf.keras.layers.Dense(2 * Vx, activation='linear')(fc3)
    output = tf.keras.layers.Reshape((Vx, 1, 2))(output)

    # define the model
    model = tf.keras.models.Model(inputs=[input_matrix_1, input_matrix_2], outputs=output)

    # compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mean_absolute_percentage_error'])

    # Train the model
    model.fit([H_input, Rx_fix_matrices_real_imag], Rx_vir_matrices_real_imag, epochs=10, batch_size=32,
              validation_split=0.2)

    # save the model
    # model.save('model_2.keras')
