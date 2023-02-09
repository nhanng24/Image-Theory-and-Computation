"""Tools for exploring image denoising.
"""
from functools import partial
import io
import logging
import os.path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize
import scipy.sparse

import PIL

logging.getLogger("PIL").setLevel(logging.ERROR)  # Suppress PIL messages

sp = scipy

plt.rcParams["image.cmap"] = "gray"  # Use greyscale as a default.


__all__ = ["subplots", "Image", "Denoise"]


def subplots(cols=1, rows=1, height=3, aspect=1, **kw):
    """More convenient subplots that automatically sets the figsize.

    Arguments
    ---------
    cols, rows : int
        Number of columns and rows in the figure.
    height : float
        Height of an individual element.  Unless specified, the figure size will be
       ``figsize=(cols * height * aspect, rows * height)``.
    aspect : float
        Aspect ratio of each figure (width/height).
    **kw : dict
        Other arguments are passed to ``plt.subplot()``
    """
    args = dict(figsize=(cols * height * aspect, rows * height))
    args.update(kw)
    return plt.subplots(rows, cols, **args)


class Base:
    """Base class for setting attributes."""

    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
        self.init()

    def init(self):
        return


class Image(Base):
    """Class to load and process images.

    Attributes
    ----------
    data : array_like | None
        If provided, then this is used as the image data.  If the first dimension has
        length 3 or 4, then it is assumed to be RGB or RGBA data respectively.  If the
        dtype is np.uint8, then it is unconverted, otherwise, it is normalized using the
        default colormap.
    dir : Path
        Directory with images.
    filename : bytes
        Location of file to be loaded if data is None.
    seed : int
        Seed for random number generator.
    """

    if os.path.exists("images"):
        # Use a local directory if it exists.  Does not need mmf_setup
        dir = Path("images")
    else:
        # Otherwise (i.e. for documentation) go relative to ROOT
        import mmf_setup

        mmf_setup.set_path()
        dir = Path(mmf_setup.ROOT) / ".." / "_data" / "images"

    filename = "The-original-cameraman-image.png"
    data = None
    seed = 2

    def __init__(self, data=None, **kw):
        self._data = data
        super().__init__(**kw)

    def init(self):
        self.rng = np.random.default_rng(seed=self.seed)
        if self._data is None:
            self._filename = Path(self.dir) / self.filename
            self.image = PIL.Image.open(self._filename)
            self.shape = self.image.size[::-1]
        else:
            data = np.asarray(self._data)
            if data.dtype == np.dtype(np.uint8):
                data = data / 255.0
            else:
                if data.max() > 1.0 or data.min() < 0:
                    data -= data.min()
                    data = data / data.max()
            self._data = data
            self.shape = data.shape

    @property
    def rgb(self):
        """Return the RGB form of the image."""
        return np.asarray(self.image.convert("RGB"))

    ######################################################################
    # Public Interface used by Denoise
    shape = None

    def get_data(self, normalize=True, sigma=0, rng=None):
        """Return greyscale image.

        Arguments
        ---------
        normalize : bool
            If `True`, then normalize the data so it is between 0 and 1.
        sigma : float
            Standard deviation (as a fraction) of gaussian noise to add to the image.
            The result will be clipped so it does not exceed (0, 255) or (0, 1) if
            `normalize==True`.
        """
        if self._data is None:
            data = np.asarray(self.image.convert("L"))
        else:
            data = self._data * 255
        vmin, vmax = 0, 255
        if normalize:
            data = data / vmax
            vmax = 1.0

        if sigma:
            if rng is None:
                rng = self.rng
            eta = sigma * rng.normal(size=data.shape)
            if normalize:
                data += eta
            else:
                data = vmax * eta + data
            data = np.minimum(np.maximum(data, vmin), vmax)
            if not normalize:
                data = data.round(0).astype("uint8")
        return data

    # End Public Interface used by Denoise
    ######################################################################

    def __repr__(self):
        return self.image.__repr__()

    def _repr_pretty_(self, *v, **kw):
        if hasattr(self, "image"):
            return self.image._repr_pretty_(*v, **kw)
        return None

    def _repr_png_(self):
        """Use the image as the representation for IPython display purposes."""
        if hasattr(self, "image"):
            return self.image._repr_png_()

        fig = self.show(self._data)
        with io.BytesIO() as png:
            fig.savefig(png)
            plt.close(fig)
            return png.getvalue()

    def show(self, u, vmin=None, vmax=None, ax=None, **kw):
        if vmax is None:
            if u.dtype == np.dtype("uint8"):
                vmax = max(255, u.max())
            else:
                vmax = max(1.0, u.max())
        if vmin is None:
            vmin = min(0, u.min())

        if ax is None:
            ax = plt.gca()

        if len(u.shape) == 1:
            ax.plot(u, **kw)
            ax.set(ylim=(vmin, vmax))
        elif len(u.shape) == 2:
            ax.imshow(u, vmin=vmin, vmax=vmax, **kw)
            ax.axis("off")
        else:
            raise NotImplementedError(f"Can't show {u.shape=}")
        fig = ax.get_figure()
        plt.close(fig)
        return fig

    imshow = show


class Denoise(Base):
    """Class for denoising images.

    Attributes
    ----------
    image : Image
        Instance of :class:`Image` with the image data.
    lam : float
        Parameter λ controlling the regularization.  Larger values will produce an image
        closer to the target.  Smaller values will produce smoother images.
    p, q : float
        Powers in the regularization term and the data fidelity terms respectively.
    use_shortcuts : bool
        If True, then use specialized shortcuts where applicable, otherwise use the
        general code.
    eps_p, eps_q : float
        Regularization constants for the regularization term and the data fidelity terms
        respectively.
    real : bool
        If True, then some operations that might be complex will return real values.
    """

    lam = 1.0
    mode = "reflect"
    image = None
    sigma = 0.5
    seed = 2
    p = 2.0
    q = 2.0

    # Compromise between performance and accuracy.
    eps_p = 1e-6  # np.finfo(float).eps
    eps_q = 1e-6  # np.finfo(float).eps

    real = True
    use_shortcuts = True

    def __init__(self, image=None, **kw):
        # Allow image to be passed as a default argument
        super().__init__(image=image, **kw)

    def init(self):
        self.rng = np.random.default_rng(seed=self.seed)
        self.u_exact = self.image.get_data(sigma=0, normalize=True)
        self.u_noise = self.image.get_data(
            sigma=self.sigma, normalize=True, rng=self.rng
        )

        # Dictionary of 1d derivative operators
        self._D1_dict = {}

        # Compute Fourier momenta
        dx = 1.0
        Nxyz = self.image.shape
        self._kxyz = np.meshgrid(
            *[2 * np.pi * np.fft.fftfreq(_N, dx) for _N in Nxyz],
            sparse=True,
            indexing="ij",
        )

        # Zero out highest momentum component.
        # for _i, _N in enumerate(Nxyz):
        #    if _N % 2 == 0:
        #        _k = np.ravel(self._kxyz[_i])
        #        _k[_N // 2] = 0
        #        self._kxyz[_i] = _k.reshape(self._kxyz[_i].shape)
        self._K2 = {
            "periodic": sum(_k**2 for _k in self._kxyz),
            "wrap": 2 * sum(1 - np.cos(_k * dx) for _k in self._kxyz) / dx**2,
        }

        # Pre-compute some energies for normalization.
        self._E_noise = self.get_energy(self.u_noise, parts=True, normalize=False)
        self._E_exact = self.get_energy(self.u_exact, parts=True)

    def _fft(self, u, axes=None, axis=None):
        # Could replace with an optimize version for pyfftw.
        if axis is None:
            return np.fft.fftn(u, axes=axes)
        else:
            return np.fft.fft(u, axis=axis)

    def _ifft(self, u, axes=None, axis=None):
        # Could replace with an optimize version for pyfftw.
        if axis is None:
            return np.fft.ifftn(u, axes=axes)
        else:
            return np.fft.ifft(u, axis=axis)

    def laplacian(self, u):
        """Return the laplacian of u."""
        if self.mode == "periodic":
            res = self._ifft(-self._K2[self.mode] * self._fft(u))
            if np.isrealobj(u):
                assert np.allclose(0, res.imag)
                res = res.real
            return res
        return sp.ndimage.laplace(u, mode=self.mode)

    def derivative1d(
        self,
        input,
        axis=-1,
        output=None,
        mode=None,
        cval=0.0,
        kind="forward",
        compute=False,
        transpose=False,
    ):
        """Return the difference of input along the specified axis.

        This is intended to be used as the `derivative` function in various
        scipy.ndimage routines.

        Arguments
        ---------
        kind : 'forward', 'backward', 'centered'
        transpose : Bool
            If True, then apply the negative transpose of the derivative operator.
        """
        if mode is None:
            mode = self.mode

        N = input.shape[axis]
        key = (N, mode, cval, kind)
        if key not in self._D1_dict:
            # Compute and add to cache
            weights = dict(
                forward=[0, -1, 1],
                backward=[-1, 1, 0],
                centered=[-0.5, 0, 0.5],
            )
            D1 = sp.sparse.lil_matrix(
                np.transpose(
                    [
                        sp.ndimage.correlate1d(
                            _I,
                            weights[kind],
                            axis=-1,
                            output=None,
                            mode=mode,
                            cval=cval,
                        )
                        for _I in np.eye(N)
                    ]
                )
            ).tocsr()
            self._D1_dict[key] = (D1, (-D1.T).tocsr())
        D1, _D1T = self._D1_dict[key]
        if transpose:
            D1 = _D1T
        if len(input.shape) > 1:
            res = np.apply_along_axis(D1.dot, axis, input)
        else:
            res = D1 @ input
        if output is not None:
            if isinstance(output, np.dtype):
                res = res.astype(output)
            else:
                output[...] = res
        return res

    def gradient(self, u, real=True):
        """Return the gradient of u.

        Arguments
        ---------
        real : bool
            If True, then return only the real part.  It is important to set this to
            False if computing the Laplacian as divergence(gradient(u)) for example.  Only
            applies for mode == "periodic".
        """
        if self.mode == "periodic":
            ut = self._fft(u)
            axes = range(1, 1 + len(u.shape))
            res = self._ifft([1j * _k * ut for _k in self._kxyz], axes=axes)
            if real and np.isrealobj(u):
                res = res.real
            return res

        u = np.asarray(u)
        du = np.array([self.derivative1d(u, axis=_i) for _i in range(len(u.shape))])
        return du

    def divergence(self, v, transpose=True):
        """Return the divergence of v."""
        v = np.asarray(v)

        if self.mode == "periodic":
            vt = self._fft(v, axes=range(1, len(v.shape)))
            res = self._ifft(sum(1j * _k * _vt for _k, _vt in zip(self._kxyz, vt)))
            if self.real:  # Can't check v here because it might be complex
                res = res.real
            return res

        return sum(
            self.derivative1d(v[_i], axis=_i, transpose=transpose)
            for _i in range(len(v.shape[1:]))
        )

    def gradient_magnitude(self, u):
        """Return the absolute magnitude of the gradient of u."""
        if self.mode == "periodic":
            return np.sqrt(self.gradient_magnitude2(u))

        return sp.ndimage.generic_gradient_magnitude(
            u, derivative=self.derivative1d, mode=self.mode
        )

    def gradient_magnitude2(self, u):
        """Return the square of the magnitude of the gradient of u."""
        return (abs(self.gradient(u, real=False)) ** 2).sum(axis=0)

    def get_energy(self, u, parts=False, normalize=False):
        """Return the energy.

        Arguments
        ---------
        parts : bool
            If True, return (E, E_regularization, E_data_fidelity)
        normalize : bool
            If True, normalize by the starting values for u_noise.
        """
        u_noise = self.u_noise
        p, q = self.p, self.q
        if p == 2.0 and self.use_shortcuts:
            E_regularization = (-u * self.laplacian(u) + self.eps_p).sum() / 2
        else:
            E_regularization = (
                (self.gradient_magnitude2(u) + self.eps_p) ** (p / 2)
            ).sum() / p

        if q == 2.0 and self.use_shortcuts:
            E_data_fidelity = ((u - u_noise) ** 2).sum() / 2
        else:
            E_data_fidelity = (((u - u_noise) ** 2 + self.eps_q) ** (q / 2)).sum() / q

        E = E_regularization + self.lam * E_data_fidelity
        E0 = self.lam * np.prod(u_noise.shape)
        if normalize:
            E0 = self._E_noise[0]

        if parts:
            return (E / E0, E_regularization / E0, E_data_fidelity / E0)
        else:
            return E / E0

    def get_denergy(self, u, normalize=False):
        """Return E'(u)."""
        u_noise = self.u_noise
        p, q = self.p, self.q

        if p == 2.0 and self.use_shortcuts:
            dE_regularization = -self.laplacian(u)
        else:
            du = self.gradient(u, real=False)
            dE_regularization = -self.divergence(
                du * (self.gradient_magnitude2(u) + self.eps_p) ** ((p - 2) / 2)
            )

        if q == 2.0 and self.use_shortcuts:
            dE_data_fidelity = u - u_noise
        else:
            dE_data_fidelity = (u - u_noise) * ((u - u_noise) ** 2 + self.eps_q) ** (
                (q - 2) / 2
            )

        dE = dE_regularization + self.lam * dE_data_fidelity

        E0 = self.lam * np.prod(u_noise.shape)
        if normalize:
            E0 = self._E_noise[0]

        return dE / E0

    def pack(self, u):
        """Return y, the 1d real representation of u for solving."""
        return np.ravel(u)

    def unpack(self, y):
        """Return `u` from the 1d real representation y."""
        return np.reshape(y, self.u_noise.shape)

    def compute_dy_dt(self, t, y):
        """Return dy_dt for the solver."""
        return -self.beta * self._df(y=y)

    def _f(self, y):
        """Return the energy"""
        return self.get_energy(self.unpack(y), normalize=True)

    def _df(self, y):
        """Return the gradient of f(y)."""
        return self.pack(self.get_denergy(self.unpack(y), normalize=True))

    def callback(self, y, plot=False):
        u = self.unpack(y)
        E, E_r, E_f = self.get_energy(u, parts=True)

        msg = f"E={E:.4g}, E_r={E_r:.4g}, E_f={E_f:.4g}"
        if plot:
            import IPython.display

            fig = plt.gcf()
            ax = plt.gca()
            ax.cla()
            IPython.display.clear_output(wait=True)
            self.image.show(u, ax=ax)
            ax.set(title=msg)
            IPython.display.display(fig)
        else:
            print(msg)

    def minimize(
        self, u0=None, method="L-BFGS-B", callback=True, tol=1e-8, plot=False, **kw
    ):
        """Directly solve the minimization problem with the L-BFGS-B method."""
        if u0 is None:
            u0 = self.u_noise
        y0 = self.pack(u0)
        if callback:
            if callback is True:
                callback = self.callback
            callback = partial(callback, plot=plot)
        res = sp.optimize.minimize(
            self._f,
            x0=y0,
            jac=self._df,
            method=method,
            callback=callback,
            tol=tol,
            **kw,
        )
        if not res.success:
            raise Exception(res.message)

        if plot:
            plt.close("all")
        u = self.unpack(res.x)
        return u

    def solve(self):
        """Directly solve the minimization problem using Fourier techniques.

        This is much faster than running `minimize` but only supports limited modes.
        """
        mode = self.mode
        if mode not in self._K2:
            raise NotImplementedError(f"{mode=} not in {set(self._K2)}")
        res = self._ifft(self._fft(self.u_noise) / (self._K2[mode] / self.lam + 1))
        if np.isrealobj(self.u_noise):
            assert np.allclose(res.imag, 0)
            res = res.real
        return res
