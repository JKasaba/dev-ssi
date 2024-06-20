import numpy as np
import pandas as pd
from scipy.optimize import minimize

# CIE 1931 color matching functions
CIE_1931_XYZ = {
    'wavelength': np.array([360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730]),
    'x_bar': np.array([0.0001299, 0.0002321, 0.0004149, 0.0007416, 0.001368, 0.002236, 0.004243, 0.00765, 0.01431, 0.02319, 0.04351, 0.07763, 0.13438, 0.21477, 0.2839, 0.3285, 0.34828, 0.34806, 0.3362, 0.3187, 0.2908, 0.2511, 0.19536, 0.1421, 0.09564, 0.05795, 0.03201, 0.0147, 0.0049, 0.0021, 0.001650001, 0.0012, 0.0011, 0.0008, 0.0006, 0.00034, 0.00024, 0.00019]),
    'y_bar': np.array([0.000003917, 0.000006965, 0.00001239, 0.00002202, 0.000039, 0.000064, 0.00012, 0.000217, 0.000396, 0.00064, 0.00121, 0.00218, 0.004, 0.0073, 0.0116, 0.01684, 0.023, 0.0298, 0.038, 0.048, 0.06, 0.0739, 0.09098, 0.1126, 0.139, 0.1693, 0.208, 0.2586, 0.323, 0.4073, 0.503, 0.6082, 0.71, 0.7932, 0.862, 0.9149, 0.954, 0.9803]),
    'z_bar': np.array([0.0006061, 0.001086, 0.001946, 0.003486, 0.006450001, 0.010550001, 0.020050001, 0.03621, 0.06785, 0.1102, 0.2074, 0.3713, 0.6456, 1.039, 1.3856, 1.62296, 1.74706, 1.7826, 1.7721, 1.7441, 1.6692, 1.5281, 1.28764, 1.0419, 0.812950001, 0.6162, 0.46518, 0.3533, 0.272, 0.2123, 0.1582, 0.1117, 0.078249999, 0.057250001, 0.04216, 0.02984, 0.0203, 0.0134])
}

def calculate_xyz(spectral_data, cmf):
    wavelengths = spectral_data['wavelength']
    intensities = spectral_data['intensity']
    x_bar = np.interp(wavelengths, cmf['wavelength'], cmf['x_bar'])
    y_bar = np.interp(wavelengths, cmf['wavelength'], cmf['y_bar'])
    z_bar = np.interp(wavelengths, cmf['wavelength'], cmf['z_bar'])
    
    X = np.sum(intensities * x_bar)
    Y = np.sum(intensities * y_bar)
    Z = np.sum(intensities * z_bar)
    
    return X, Y, Z

def calculate_chromaticity(X, Y, Z):
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    return x, y

def planckian_locus_approximation(u, v):
    # Approximation of the Planckian locus for color temperature calculation
    return (-0.2661239 * (10 ** 9) / (u ** 3)) - (0.2343580 * (10 ** 6) / (u ** 2)) + (0.8776956 * (10 ** 3) / u) + 0.179910

def calculate_cct_from_uv(u, v):
    # Use the inverse of the approximation to find the color temperature
    def objective_function(T):
        u_approx = (-0.2661239 * (10 ** 9) / (T ** 3)) - (0.2343580 * (10 ** 6) / (T ** 2)) + (0.8776956 * (10 ** 3) / T) + 0.179910
        return (u - u_approx) ** 2

    result = minimize(objective_function, x0=[6500], bounds=[(1000, 40000)])
    return result.x[0]

def calculate_cct(spectral_data):
    X, Y, Z = calculate_xyz(spectral_data, CIE_1931_XYZ)
    x, y = calculate_chromaticity(X, Y, Z)
    u = 4 * x / (-2 * x + 12 * y + 3)
    v = 9 * y / (-2 * x + 12 * y + 3)
    return calculate_cct_from_uv(u, v)

# Example usage with spectral data
spectral_data = {
    'wavelength': np.array([360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730]),
    'intensity': np.random.random(38)  # Replace with actual intensity data
}

cct_value = calculate_cct(spectral_data)
print(f"CCT: {cct_value} K")
