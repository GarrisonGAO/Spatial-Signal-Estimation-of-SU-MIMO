import numpy as np
import cmath
import matplotlib.pyplot as plt

# set the print of complex matrix
np.set_printoptions(precision=2, suppress=True, linewidth=100, formatter={'complexfloat': '{:+.1f}'.format})

from itertools import product
import tensorflow as tf
import h5py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# we assume the SNR is 30dB
def generate_complex_gaussian_noise(N, mean=0, variance=0.001):
    # get the standard derivation
    std = np.sqrt(variance)

    # generate the real and imaginary parts
    real_part = np.random.normal(mean, std, N)
    imaginary_part = np.random.normal(mean, std, N)

    # generate the complex gaussian matrix
    complex_noise = real_part + 1j * imaginary_part

    # Turn it to be N x 1 matrix
    complex_noise_matrix = complex_noise.reshape(N, 1)

    return complex_noise_matrix


def calculate_ber(transmitted_bits, received_bits):
    # calculate the number of different bit
    error_bits = sum(t != r for t, r in zip(transmitted_bits, received_bits))

    # calculate the total bit number
    total_bits = len(transmitted_bits)

    # Calculate the BER
    ber = error_bits / total_bits

    return ber


def qpsk_symbol_to_bit(qpsk_vector):
    # Initialize one bit stream array
    bit_stream = []

    for symbol in qpsk_vector:
        real = symbol.real
        imag = symbol.imag

        # decide the symbol and its corresponding bits
        if real >= 0 and imag >= 0:
            bit_stream.extend([0, 0])
        elif real < 0 and imag >= 0:
            bit_stream.extend([0, 1])
        elif real < 0 and imag < 0:
            bit_stream.extend([1, 0])
        elif real >= 0 and imag < 0:
            bit_stream.extend([1, 1])

    return bit_stream


def ml_estimation(H, Y):
    # Define the QPSK symbols
    qpsk_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

    Nt = H.shape[1]  # number of Tx antenna

    # generate all the possible Tx signal combination
    all_combinations = np.array(list(product(qpsk_symbols, repeat=Nt)))

    best_combination = None
    min_distance = float('inf')

    # go through all the signal combination
    for X in all_combinations:
        X = X.reshape(-1, 1)  # adjust to column signal
        Y_hat = np.dot(H, X)  # calculate the received signal
        distance = np.linalg.norm(Y - Y_hat)  # calculate the distance between estimated signal and real signal
        if distance < min_distance:
            min_distance = distance
            best_combination = X

    return best_combination


def mmse_detection(H, y, noise_variance=0.001):
    """
    MMSE detection algorithm.

    Parameters:
    H (np.ndarray): Channel matrix (MxN)
    y (np.ndarray): Received signal vector (Mx1)
    noise_variance (float): Noise variance

    Returns:
    np.ndarray: Estimated transmitted signal vector (Nx1)
    """
    # Compute the regularization parameter
    lambda_ = noise_variance

    # Compute the MMSE estimator
    H_H = H.conj().T
    regularization_term = lambda_ * np.eye(H.shape[1])
    H_H_H = np.dot(H_H, H)
    inverse_term = np.linalg.inv(H_H_H + regularization_term)
    H_H_y = np.dot(H_H, y)
    x_hat = np.dot(inverse_term, H_H_y)

    return x_hat


def zero_forcing_equalizer(H, Y):
    """
    Zero Forcing Equalizer implementation.

    Parameters:
    H: The channel matrix.
    Y: The received signal vector.

    Returns: The estimated transmitted signal vector.
    """
    # Compute the pseudo-inverse of the channel matrix H
    H_pseudo_inverse = np.linalg.pinv(H)

    # Estimate the transmitted signal using Zero Forcing
    X_hat = np.dot(H_pseudo_inverse, Y)

    return X_hat


def QPSK(N, path_number):
    # Generate an array of random integers from 1 to 4
    int_symbols = np.random.randint(1, 5, N)

    # Initialize an array for QPSK symbols
    qpsk_symbols = np.zeros(N, dtype=complex)

    # Map the integers to QPSK symbols
    qpsk_symbols[int_symbols == 1] = 1 + 1j
    qpsk_symbols[int_symbols == 2] = 1 - 1j
    qpsk_symbols[int_symbols == 3] = -1 + 1j
    qpsk_symbols[int_symbols == 4] = -1 - 1j

    # Normalize the output energy
    qpsk_symbols = qpsk_symbols / np.sqrt(2)
    qpsk_symbols = qpsk_symbols.reshape((N, 1))

    return qpsk_symbols


def effective_rank(A):
    # GET THE MATRIX SINGULAR VALUES
    singular_values = np.linalg.svd(A, compute_uv=False)
    # We only want the non-zero singular values
    non_zero_singular = singular_values[singular_values > 0]
    # Calculate the p
    p = non_zero_singular / np.sum(non_zero_singular)
    # Calculate the Shannon Entropy (make sure it could work when we have singular values = 0)
    entropy = -np.sum(p * np.log(p))
    # return the effective rank
    return np.exp(entropy)


def signal_energy(complex_matrix):
    # Calculate the energy of each symbol
    energy = np.abs(complex_matrix) ** 2

    # Calculate the average energy
    average_energy = np.mean(energy)

    return average_energy


def Tx_steering_vector(Nt, Dt, radians):
    cos_theta = np.cos(radians)
    k_values = np.arange(1, Nt + 1)
    exponent = -1j * 2 * np.pi * (k_values - 1) * Dt * cos_theta
    vector = np.exp(exponent)
    matrix = vector.reshape((Nt, 1))
    return matrix


def Rx_steering_vector(Nr, Nr_virtual, Dr, radians):
    cos_theta = np.cos(radians)

    # Positions of the fixed Rx antennas
    k_values1 = np.arange(1, Nr + 1)

    # Positions of the virtual Rx antennas, where they are inserted between the fixed Rx antennas
    k_values2 = np.arange(2.5, Nr_virtual + 2.5, 1)

    # Combine the positions of fixed and virtual antennas
    k_values = np.concatenate((k_values1, k_values2))

    # print(k_values)
    exponent_big = -1j * 2 * np.pi * (k_values - 1) * Dr * cos_theta
    exponent_small = -1j * 2 * np.pi * (k_values1 - 1) * Dr * cos_theta

    vector_big = np.exp(exponent_big)
    vector_small = np.exp(exponent_small)

    matrix_big = vector_big.reshape((Nr + Nr_virtual, 1))
    matrix_small = vector_small.reshape((Nr, 1))

    return matrix_big, matrix_small


def multipath_channel(l_total, Nt):
    L = l_total - 1
    alpha = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)

    # Insert the LoS component (1, we normalize the elements) in a random position
    random_position = np.random.randint(0, len(alpha) + 1)
    arr_los = np.insert(alpha, random_position, 1)

    # # considering the number of paths
    arr_los = arr_los / np.sqrt(l_total) / np.sqrt(Nt)

    # Generate the diagonal matrix
    diagonal_matrix = np.diag(arr_los)
    return diagonal_matrix


def channel_matrix_has_LOS(Dr, Dt, L_total, Nr, Nt, Nr_virtual):
    # AoA, AoD and fading matrix (diagonal matrix) will be generated randomly
    # the position parameters of receiver and transmitter are all fixed

    # Firstly, decide the fading matrix In this multipath far field scenario, ignore the effect of path loss,
    diagonal_matrix = multipath_channel(L_total, Nt)

    # Secondly, is the receiver response matrix
    radians_AoA = np.random.uniform(0, 2 * np.pi, size=L_total)
    # Pile along the horizontal direction
    matrixR_big = np.hstack([Rx_steering_vector(Nr, Nr_virtual, Dr, radian)[0] for radian in radians_AoA])
    matrixR_small = np.hstack([Rx_steering_vector(Nr, Nr_virtual, Dr, radian)[1] for radian in radians_AoA])

    # Thirdly, is the transmitter response array
    radians_AoD = np.random.uniform(0, 2 * np.pi, size=L_total)
    matrixT = np.hstack([Tx_steering_vector(Nt, Dt, radian) for radian in radians_AoD])

    H_big = matrixR_big @ diagonal_matrix @ np.conjugate(np.transpose(matrixT))
    H_small = matrixR_small @ diagonal_matrix @ np.conjugate(np.transpose(matrixT))
    return H_big, H_small


# normalization matrix：turn the input data between [0, 1]
def normalize_data(data):
    data_min = np.min(data, axis=(1, 2, 3), keepdims=True)
    data_max = np.max(data, axis=(1, 2, 3), keepdims=True)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data


def symbol_estimation(H_total_fix_vir, Rx_symbol_fix, model):
    # extract the shape of Rx,Vx 和 Tx
    Rx = Rx_symbol_fix.shape[0]
    Vx = H_total_fix_vir.shape[0] - Rx_symbol_fix.shape[0]
    Tx = H_total_fix_vir.shape[1]

    # take pre-process
    # transform the complex matrix from A*B into A*B*2, a double channel matrix
    Rx_symbol_fix_input = np.stack((Rx_symbol_fix.real, Rx_symbol_fix.imag), axis=-1)
    H_total_fix_vir_input = np.stack((H_total_fix_vir.real, H_total_fix_vir.imag), axis=-1)

    # based on the original three dimensions, add one more dimension
    Rx_symbol_fix_input = Rx_symbol_fix_input.reshape((1, Rx, 1, 2))
    H_total_fix_vir_input = H_total_fix_vir_input.reshape((1, Rx + Vx, Tx, 2))

    # Padding the Rx_fix_matrices_real with zeros
    # combine the padded Rx and H_matries_real_imag，get H_input
    Rx_add_zeros = np.zeros((1, Rx + Vx, 1, 2))
    Rx_add_zeros[:, :Rx, :, :] = Rx_symbol_fix_input
    H_input = np.concatenate((Rx_add_zeros, H_total_fix_vir_input), axis=2)

    # Normalize the matrix
    H_input = normalize_data(H_input)
    Rx_symbol_fix_input = normalize_data(Rx_symbol_fix_input)

    # input H and RX, estimate Vx signal
    Rx_symbol_vir_out_hat = model.predict([H_input, Rx_symbol_fix_input])

    # remove the first dimension of output
    Rx_symbol_vir_out_hat = Rx_symbol_vir_out_hat.reshape((2, 1, 2))
    Rx_real = Rx_symbol_vir_out_hat[..., 0]
    Rx_imag = Rx_symbol_vir_out_hat[..., 1]

    # turn Vx signal into complex signal
    Rx_symbol_vir_hat = Rx_real + 1j * Rx_imag

    # combine Rx_symbol_fix and Rx_symbol_vir，get Rx_hat
    Rx_hat = np.vstack((Rx_symbol_fix, Rx_symbol_vir_hat))
    return Rx_hat


# Define the parameters of the transmitter and receiver antennas
Dr = 0.1  # inter-distance in the receiver array
Nr = 7  # antenna number of receiver array
Nr_virtual = 2
Dt = 0.5  # inter-distance in the transmitter array
Nt = 8  # antenna number of transmitter array
L_total = 5  # number of scatters

model = tf.keras.models.load_model('model_2.keras')

ber_list_vir = []
ber_list_fix = []
ber_list = []

for _ in range(1000):
    # get the input， Rx_symbol_fix and H_total_fix_vir
    H_total_fix_vir, H_total_fix = channel_matrix_has_LOS(Dr=Dr, Dt=Dt, L_total=L_total, Nr=Nr, Nt=Nt,
                                                          Nr_virtual=Nr_virtual)
    Tx_symbol = QPSK(Nt, L_total)
    noise = generate_complex_gaussian_noise(N=Nr + Nr_virtual, mean=0, variance=0.01)
    Rx_symbol = H_total_fix_vir @ Tx_symbol + noise

    Rx_symbol_fix = Rx_symbol[:Nr]
    Rx_symbol_vir = Rx_symbol[-Nr_virtual:]

    # Estimate the symbols which received by the symbol antenna
    Rx_hat = symbol_estimation(H_total_fix_vir, Rx_symbol_fix, model)

    # Rx_hat and H_total_fix_vir take MMSE， get Tx_SE_hat, under the condition with virtual antenna
    Tx_vir_hat = mmse_detection(H_total_fix_vir, Rx_hat, noise_variance=0.01)
    #Tx_vir_hat = ml_estimation(H_total_fix_vir, Rx_hat)

    # Rx_symbol_fix and H_total_fix take MMSE， get Tx_fix_hat, under the condition without virtual antenna
    Tx_fix_hat = mmse_detection(H_total_fix, Rx_symbol_fix, noise_variance=0.01)
    #Tx_vir_hat = ml_estimation(H_total_fix_vir, Rx_hat)

    # assume could get the completely correct Vx values, under the condition with ideal virtual antenna
    Tx_hat = mmse_detection(H_total_fix_vir, Rx_symbol, noise_variance=0.01)

    # real bit stream
    Tx_bit_stream = qpsk_symbol_to_bit(Tx_symbol)

    # bit stream with virtual antenna
    Tx_vir_bit_stream = qpsk_symbol_to_bit(Tx_vir_hat)

    # bit stream with fixed antenna, without virtual antenna
    Tx_fix_bit_stream = qpsk_symbol_to_bit(Tx_fix_hat)

    # Ideal virtual antenna
    Tx_ideal_vir_stream = qpsk_symbol_to_bit(Tx_hat)

    ber_vir = calculate_ber(transmitted_bits=Tx_bit_stream, received_bits=Tx_vir_bit_stream)
    ber_fix = calculate_ber(transmitted_bits=Tx_bit_stream, received_bits=Tx_fix_bit_stream)
    ber = calculate_ber(transmitted_bits=Tx_bit_stream, received_bits=Tx_ideal_vir_stream)

    ber_list_vir.append(ber_vir)
    ber_list_fix.append(ber_fix)
    ber_list.append(ber)

# compare the BER of:
# SU-MIMO with practical virtual antenna
# SU-MIMO with ideal virtual antenna
# SU-MIMO without virtual antenna


average_ber_vir = np.mean(ber_list_vir)
average_ber_fix = np.mean(ber_list_fix)
average_ber = np.mean(ber_list)
print("The average ber with virtual antenna is", average_ber_vir)
print("The average ber with ideal virtual antenna is", average_ber)
print("The average ber without virtual antenna is", average_ber_fix)
