import tensorflow as tf
import h5py
import numpy as np
import keras
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers

np.set_printoptions(precision=2, suppress=True, linewidth=100, formatter={'complexfloat': '{:+.2f}'.format})

print(keras.__version__)


# normalization function, [0, 1]
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

    # take the samples
    sample_number = 10000
    with h5py.File(file_path, 'r') as f:
        H_matrices = f['H'][:sample_number]
        Rx_fix_matrices = f['Rx_fix'][:sample_number]
        Rx_vir_matrices = f['Rx_vir'][:sample_number]

    # transform the complex matrix into the double channel matrix
    H_matrices_real_imag = np.stack((H_matrices.real, H_matrices.imag), axis=-1)
    Rx_fix_matrices_real_imag = np.stack((Rx_fix_matrices.real, Rx_fix_matrices.imag), axis=-1)
    Rx_vir_matrices_real_imag = np.stack((Rx_vir_matrices.real, Rx_vir_matrices.imag), axis=-1)

    # Padding the Rx_fix_matrices with zero
    # Combine the padded Rx and H_matries_real_imag，get H_input
    Rx_add_zeros = np.zeros((sample_number, Rx + Vx, 1, 2))
    Rx_add_zeros[:, :Rx, :, :] = Rx_fix_matrices_real_imag
    H_input = np.concatenate((Rx_add_zeros, H_matrices_real_imag), axis=2)

    # Normalize all the double-channel inputs
    H_input = normalize_data(H_input)
    Rx_fix_matrices_real_imag = normalize_data(Rx_fix_matrices_real_imag)

    # build the model, define the input layer
    # First input layer received the combined H matrix
    # Second input layer received the Rx vector
    L2_parameter = 0.0001
    input_matrix_1 = tf.keras.layers.Input(shape=(Rx + Vx, Tx + 1, 2), name='input_image_1')
    input_matrix_2 = tf.keras.layers.Input(shape=(Rx, 1, 2), name='input_image_2')

    # In the first input layer, take convolutional operation over channel matrix H
    x1 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(
        input_matrix_1)
    x1 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x1)
    x1 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x1)
    x1 = tf.keras.layers.Flatten(name="x1_flatten")(x1)
    x1 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x1)

    # In the second input layer, take convolutional operation over Rx_fix
    x2 = tf.keras.layers.Conv2D(32, (3, 1), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(L2_parameter))(input_matrix_2)
    x2 = tf.keras.layers.Conv2D(64, (3, 1), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(L2_parameter))(x2)
    x2 = tf.keras.layers.Conv2D(128, (3, 1), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(L2_parameter))(x2)
    x2 = tf.keras.layers.Flatten(name="x2_flatten")(x2)
    x2 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x2)

    # combine the 2 outputs, but multiply them
    merged = tf.keras.layers.Multiply(name="merge_multiply")([x1, x2])

    # MLP
    f = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter), name="F1")(
        merged)
    f = tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(L2_parameter), name="F2")(f)
    f = tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(L2_parameter), name="F3")(f)

    # Output layer
    output = tf.keras.layers.Dense(2 * Vx, activation='linear', name="output")(f)
    output = tf.keras.layers.Reshape((Vx, 1, 2), name="output_reshape")(output)

    # define the model
    model = tf.keras.models.Model(inputs=[input_matrix_1, input_matrix_2], outputs=output)

    # compile model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse',
                  metrics=['mean_absolute_percentage_error'])

    # Train model
    model.fit([H_input, Rx_fix_matrices_real_imag], Rx_vir_matrices_real_imag, epochs=10, batch_size=32,
              validation_split=0.2)

    # # save model
    # model.save('model_3.keras')
