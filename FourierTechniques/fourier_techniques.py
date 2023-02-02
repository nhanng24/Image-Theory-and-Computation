import numpy as np
import scipy.ndimage
import scipy.optimize
import matplotlib.pyplot as plt

sp = scipy

_EPS = np.finfo(float).eps

__all__ = ["np", "sp", "plt", "L", "f", "D1_f", "D2_f"]

L = 10.0


def f(x, eta=1, d=0, n=1, L=L):
    """Return the dth derivative of f(x)."""
    k = 2 * np.pi * n / L
    c = np.cos(k * x)
    eta_c_1 = eta * c + 1 + _EPS
    f = np.exp(-1 / eta_c_1)
    if d == 0:
        res = f
    elif d == 1:
        res = -eta * k / eta_c_1**2 * np.sin(k * x) * f
    elif d == 2:
        c2 = c**2
        c3 = c * c2
        res = -eta * k**2 / eta_c_1**4 * (eta * (1 + c2 - eta * c3) +
                                          (1 + 2 * eta**2) * c) * f
    return res


def D1_f(f, x):
    """Compute the derivative of f using Fourier methods."""
    dx = np.diff(x).mean()
    N = len(f)
    kx = 2 * np.pi * np.fft.fftfreq(N, dx)
    df = np.fft.ifft(1j * kx * np.fft.fft(f))
    #assert np.allclose(df.imag, 0)
    return df.real


def D2_f(f, x):
    """Compute the second derivative of f using Fourier methods."""
    dx = np.diff(x).mean()
    N = len(f)
    kx = 2 * np.pi * np.fft.fftfreq(N, dx)
    df = np.fft.ifft(-kx**2 * np.fft.fft(f))
    #assert np.allclose(df.imag, 0)
    return df.real
