import tensorflow as tf
import h5py
import numpy as np
import keras
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, activations

np.set_printoptions(precision=2, suppress=True, linewidth=100, formatter={'complexfloat': '{:+.2f}'.format})


# Normalize into [0, 1]
def normalize_data(data):
    data_min = np.min(data, axis=(1, 2, 3), keepdims=True)
    data_max = np.max(data, axis=(1, 2, 3), keepdims=True)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

# residual_block, to avoid the vanishing gradient problem
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activations.gelu(x)
    x = layers.add([x, shortcut])
    return x


class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        # define the Feed Forward Layer (MLP Block)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation=None),  # Linear
            layers.Activation('gelu'),  # GELU activation
            layers.Dropout(rate),  # Dropout
            layers.Dense(key_dim),  # Linear
            layers.Dropout(rate)  # Dropout
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        # First normalization
        norm1 = self.layernorm1(inputs)
        # Attention layer
        attn_output = self.attention(norm1, norm1)
        # Dropout
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection
        out1 = inputs + attn_output

        # Second normalization
        norm2 = self.layernorm2(out1)
        # Feed Forward Network
        ffn_output = self.ffn(norm2)
        # Dropout
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection
        return out1 + ffn_output


Tx = 8
Rx = 7
Vx = 2

# Define Dr, inter-distance in the receiver array
# Dr change from 0.1 to 0.5，inter-spacing is 0.1
for Dr in np.arange(0.1, 0.6, 0.1):

    print("The Dr is", Dr)

    file_path = f"/Sample(7+2_8)_Dr={Dr:.1f}_n=0.01_10e5.h5"

    # Extract Samples
    sample_number = 10000
    with h5py.File(file_path, 'r') as f:
        H_matrices = f['H'][:sample_number]
        Rx_fix_matrices = f['Rx_fix'][:sample_number]
        Rx_vir_matrices = f['Rx_vir'][:sample_number]

    # Transform H, Rx_fix and Rx_vir from complex matrices intro double channel matrices
    H_matrices = np.stack((H_matrices.real, H_matrices.imag), axis=-1)
    Rx_fix_matrices = np.stack((Rx_fix_matrices.real, Rx_fix_matrices.imag), axis=-1)
    Rx_vir_matrices = np.stack((Rx_vir_matrices.real, Rx_vir_matrices.imag), axis=-1)

    # Padding Rx_fix_matrices with zeros
    Rx_add_zeros = np.zeros((sample_number, Rx + Vx, 1, 2))
    Rx_add_zeros[:, :Rx, :, :] = Rx_fix_matrices

    # # Normalize all the input matrices
    # H_matrices = normalize_data(H_matrices)
    # Rx_add_zeros = normalize_data(Rx_add_zeros)

    num_heads = 4
    em_dim = 256
    ff_dim = 128
    num_encoder_layers = 6
    L2_parameter = 0.0001

    # Build model, has 2 input layers
    # Embedding the H
    input_matrix_1 = layers.Input(shape=(Rx + Vx, Tx, 2), name='input_image_1')
    x1 = layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(input_matrix_1)
    x1 = layers.Conv2D(128, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x1)
    x1 = layers.Flatten(name="x1_flatten")(x1)

    x1 = layers.Dense((Rx + Vx) * em_dim, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x1)
    x1 = layers.Reshape((Rx + Vx, em_dim))(x1)  # Reshape the output to (9, 256)

    # Embedding the Rx_add_zeros
    input_matrix_2 = layers.Input(shape=(Rx + Vx, 1, 2), name='input_image_2')
    x2 = layers.Conv2D(128, (2, 1), activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(input_matrix_2)
    x2 = layers.Flatten(name="x2_flatten")(x2)

    x2 = layers.Dense((Rx + Vx) * em_dim, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x2)
    x2 = layers.Reshape((Rx + Vx, em_dim))(x2)  # Reshape the output to (9, 256)

    # 构建可训练的 positional embedding
    positional_embedding = tf.Variable(initial_value=tf.random.normal([Rx + Vx, em_dim]), trainable=True,
                                       name="positional_embedding")

    # Add x1, x2 and positional embedding
    combine = x1 + x2 + positional_embedding

    # Transformer Encoder
    # Composed by transformer encoder blocks
    x = TransformerEncoder(num_heads=num_heads, key_dim=em_dim, ff_dim=ff_dim)(combine)
    for _ in range(num_encoder_layers - 1):
        x = TransformerEncoder(num_heads=num_heads, key_dim=em_dim, ff_dim=ff_dim)(x)

    # Transformer Decoder
    # Composed by residual blocks
    x = layers.Reshape((9, 256, 1))(x)  # add one more dimension
    x = layers.Conv2D(filters=32, kernel_size=(1, 8), strides=(1, 8), padding='valid')(x)
    # 5 residual blocks
    for _ in range(5):
        x = residual_block(x, filters=32)
    x = layers.Conv2D(filters=2, kernel_size=(3, 4), strides=(3, 4), padding='valid')(x)

    # The final output part
    x = layers.Flatten()(x)
    x = layers.Dense(24, activation='relu', kernel_regularizer=regularizers.l2(L2_parameter))(x)
    x = layers.Dropout(0.1)(x)
    # Add 1 normalization layer
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(12, activation='softmax', kernel_regularizer=regularizers.l2(L2_parameter))(x)
    x = layers.Dropout(0.1)(x)
    # Add one normalization layer
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(4, activation='softmax', kernel_regularizer=regularizers.l2(L2_parameter))(x)
    output = layers.Reshape((2, 1, 2))(x)

    # Define the model
    model = tf.keras.models.Model(inputs=[input_matrix_1, input_matrix_2], outputs=output)

    # compile model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse',
                  metrics=['mean_absolute_percentage_error'])

    # Train the model
    model.fit([H_matrices, Rx_add_zeros], Rx_vir_matrices, epochs=8, batch_size=32, validation_split=0.2)

    # # save model
    # model.save('model_Rx=7+2_Tx=8_Dr=0.5_n=0.01_2.h5')
