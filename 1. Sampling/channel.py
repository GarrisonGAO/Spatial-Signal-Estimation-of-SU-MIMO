import numpy as np
import cmath

import matplotlib.pyplot as plt

# set the print of complex matrix
np.set_printoptions(precision=2, suppress=True, linewidth=100, formatter={'complexfloat': '{:+.1f}'.format})

from itertools import product
import h5py


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

    Nt = H.shape[1]  # 发射端天线数

    # Generate all the possible signal combination
    all_combinations = np.array(list(product(qpsk_symbols, repeat=Nt)))

    best_combination = None
    min_distance = float('inf')

    # Go through all the signal combinations
    for X in all_combinations:
        X = X.reshape(-1, 1)  # adjust the column vector
        Y_hat = np.dot(H, X)  # 计算接收信号
        distance = np.linalg.norm(Y - Y_hat)  # calculate distance to the expected signal
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
    k_values2 = np.arange(3.5, Nr_virtual + 3.5, 1)

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
    # Nt is useful in normalization
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


# Define the parameters of the transmitter and receiver antennas
Dr = 0.5  # inter-distance in the receiver array
Nr = 7  # antenna number of receiver array
Nr_virtual = 2
Dt = 0.5  # inter-distance in the transmitter array
Nt = 8  # antenna number of transmitter array
L_total = 10  # number of scatters

# Generate 100000 set of samples
num_samples = 10000 * 10
H_matrices = []
Rx_fix_matrices = []
Rx_vir_matrices = []

for _ in range(num_samples):
    # inputs:
    # Rx_symbol_fix and H_total_fix_vir

    # output:
    # Rx_symbol_vir

    H_total_fix_vir, H_total_fix = channel_matrix_has_LOS(Dr=Dr, Dt=Dt, L_total=L_total, Nr=Nr, Nt=Nt,
                                                          Nr_virtual=Nr_virtual)
    Tx_symbol = QPSK(Nt, L_total)
    noise = generate_complex_gaussian_noise(N=Nr + Nr_virtual, mean=0, variance=0.01)
    Rx_symbol_fix_vir = H_total_fix_vir @ Tx_symbol + noise

    Rx_symbol_fix = Rx_symbol_fix_vir[:Nr]
    Rx_symbol_vir = Rx_symbol_fix_vir[-Nr_virtual:]

    H_matrices.append(H_total_fix_vir)
    Rx_fix_matrices.append(Rx_symbol_fix)
    Rx_vir_matrices.append(Rx_symbol_vir)

# save the samples .h5 文件中
with h5py.File('Sample(7+2_8)_Dr=0.5_n=0.01_10e5.h5', 'w') as f:
    f.create_dataset('H', data=np.array(H_matrices))
    f.create_dataset('Rx_fix', data=np.array(Rx_fix_matrices))
    f.create_dataset('Rx_vir', data=np.array(Rx_vir_matrices))
print("It has been stored：0.5")
