"""Tools for exploring image denoising.
"""
from collections import namedtuple
from functools import partial
import io
import itertools
import logging
import math
import os.path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.collections import LineCollection
from matplotlib import cm

import scipy.ndimage
import scipy.optimize
import scipy.sparse

from tqdm.auto import tqdm

import PIL

try:
    import maxflow
except ImportError:
    maxflow = None

logging.getLogger("PIL").setLevel(logging.ERROR)  # Suppress PIL messages

sp = scipy

plt.rcParams["image.cmap"] = "gray"  # Use greyscale as a default.

__all__ = ["subplots", "Image", "Denoise", "L1TV", "L1TVMaxFlow"]

_EPS = np.finfo(float).eps


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
    copy : bool
        If True, make a copy of the image.
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

    skimages = {
        "astronaut",
        "binary_blobs",
        "brick",
        "colorwheel",
        "camera",
        "cat",
        "checkerboard",
        "clock",
        "coffee",
        "coins",
        "eagle",
        "grass",
        "gravel",
        "horse",
        "logo",
        "page",
        "text",
        "rocket",
    }

    def __init__(self, data=None, copy=True, **kw):
        if data is None:
            self._data = None
        elif isinstance(data, str) and data in self.skimages:
            import skimage  # Soft dependency

            self._data = getattr(skimage.data, data)()
        elif copy:
            self._data = np.array(data)
        else:
            self._data = np.asarray(data)
        super().__init__(**kw)

    def init(self):
        self.rng = np.random.default_rng(seed=self.seed)
        if self._data is None:
            self._filename = Path(self.dir) / self.filename
            self.image = PIL.Image.open(self._filename)
            self.shape = self.image.size[::-1]
        else:
            data = np.asarray(self._data)
            dtype = data.dtype
            if len(data.shape) == 3:
                # Color image, make greyscale.
                data = data[..., :3].mean(axis=-1)
                if dtype == np.dtype(np.uint8):
                    data = np.round(data, 0).astype(dtype)
            if dtype == np.dtype(np.uint8):
                data = data / 255.0
            else:
                if data.max() > 1.0 or data.min() < 0:
                    data -= data.min()
                    data = data / data.max()
            self._data = data
            self.shape = data.shape

    def dist(self, u, u_exact=None):
        """Return our canonical distance measure: mean of the square of the deviations.

        Arguments
        ---------
        u : array-like
            The image to compare.
        u_exact : array-like, None
            The exact answer.  Will use self.get_data() if None.
        """
        if u_exact is None:
            u_exact = self.get_data()
        return ((u - u_exact)**2).mean()

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

    def show(
        self,
        u,
        u_noise=None,
        u_exact=None,
        *v,
        titles=None,
        vmin=None,
        vmax=None,
        ax=None,
        height=3,
        **kw,
    ):
        """Show the image u.

        Arguments
        ---------
        u, u_noise, u_exact, *v : array-like
            Images to be shown.
            axes will be allocated with u on the left.
        vmin, vmax : float, None
            If provided, then these will be provided to
            :func:`matplotlib.axes.Axes.imshow`, otherwise, they will be computed as
            follows.  If ``u.dtype`` is ``uint8``, then  ``vmax=255``, otherwise
            ``vmax=1``.  ``vmin=min(0, u.min())``.  The right-most image will be used to
            compute vmin, vmax.
        titles : [str], None
            Titles for axes.
        ax : Axes, None
            If provided, then the image will be drawn in this axes instance.
        height : float
            Passed to :func:`subplots` if ax is None.
        **kw : {}
            Any additional arguments will be passed through to
            :func:`matplotlib.axes.Axes.imshow`.

        """
        dim = len(u.shape)
        us = [u]
        title_dict = {}
        if u_noise is not None:
            us.append(u_noise)
            if len(v) == 0:
                title_dict[0] = "u"
                title_dict[len(us) - 1] = "u_noise"
        if u_exact is not None:
            us.append(u_exact)
            if len(v) == 0:
                title_dict[0] = "u"
                title_dict[len(us) - 1] = "u_exact"

        if titles:
            title_dict = dict(enumerate(titles))

        us.extend(v)
        if len(us) == 1:
            if ax is None:
                ax = plt.gca()
            axs = [ax]
            fig = ax.figure
        else:
            if ax is None:
                if dim == 2:
                    aspect = np.divide(*u.shape)
                else:
                    aspect = 1
                fig, axs = subplots(len(us), height=height, aspect=aspect)
            else:
                gs = GridSpecFromSubplotSpec(1,
                                             len(us),
                                             subplot_spec=ax.get_subplotspec())
                ax.set_subplotspec(gs[0])
                fig = ax.figure
                axs = [ax] + list(map(fig.add_subplot, list(gs)[1:]))

        if vmax is None:
            if us[-1].dtype == np.dtype("uint8"):
                vmax = max(255, us[-1].max())
            else:
                vmax = max(1.0, us[-1].max())
        if vmin is None:
            vmin = min(0, us[-1].min())

        for _n, (_u, _ax) in enumerate(zip(us, axs)):
            if dim == 1:
                _ax.plot(_u, **kw)
                _ax.set(ylim=(vmin, vmax))
            elif dim == 2:
                _ax.imshow(_u, vmin=vmin, vmax=vmax, **kw)
                _ax.axis("off")
            else:
                raise NotImplementedError(f"Can't show {_u.shape=}")

            _ax.set(title=title_dict.get(_n, None))
        plt.sca(axs[0])
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
    sigma : float
        Standard deviation (as a fraction) of gaussian noise to add to the image.
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
        self.u_noise = self.image.get_data(sigma=self.sigma,
                                           normalize=True,
                                           rng=self.rng)

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
        self._E_noise = self.get_energy(self.u_noise,
                                        parts=True,
                                        normalize=False)
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
                np.transpose([
                    sp.ndimage.correlate1d(
                        _I,
                        weights[kind],
                        axis=-1,
                        output=None,
                        mode=mode,
                        cval=cval,
                    ) for _I in np.eye(N)
                ])).tocsr()
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
        du = np.array(
            [self.derivative1d(u, axis=_i) for _i in range(len(u.shape))])
        return du

    def divergence(self, v, transpose=True):
        """Return the divergence of v."""
        v = np.asarray(v)

        if self.mode == "periodic":
            vt = self._fft(v, axes=range(1, len(v.shape)))
            res = self._ifft(
                sum(1j * _k * _vt for _k, _vt in zip(self._kxyz, vt)))
            if self.real:  # Can't check v here because it might be complex
                res = res.real
            return res

        return sum(
            self.derivative1d(v[_i], axis=_i, transpose=transpose)
            for _i in range(len(v.shape[1:])))

    def gradient_magnitude(self, u):
        """Return the absolute magnitude of the gradient of u."""
        if self.mode == "periodic":
            return np.sqrt(self.gradient_magnitude2(u))

        return sp.ndimage.generic_gradient_magnitude(
            u, derivative=self.derivative1d, mode=self.mode)

    def gradient_magnitude2(self, u):
        """Return the square of the magnitude of the gradient of u."""
        return (abs(self.gradient(u, real=False))**2).sum(axis=0)

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
                (self.gradient_magnitude2(u) + self.eps_p)**(p / 2)).sum() / p

        if q == 2.0 and self.use_shortcuts:
            E_data_fidelity = ((u - u_noise)**2).sum() / 2
        else:
            E_data_fidelity = ((
                (u - u_noise)**2 + self.eps_q)**(q / 2)).sum() / q

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
                du * (self.gradient_magnitude2(u) + self.eps_p)**((p - 2) / 2))

        if q == 2.0 and self.use_shortcuts:
            dE_data_fidelity = u - u_noise
        else:
            dE_data_fidelity = (u - u_noise) * (
                (u - u_noise)**2 + self.eps_q)**((q - 2) / 2)

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

    def minimize(self,
                 u0=None,
                 method="L-BFGS-B",
                 callback=True,
                 tol=1e-8,
                 plot=False,
                 **kw):
        """Directly solve the minimization problem with the L-BFGS-B method."""
        if u0 is None:
            u0 = self.u_noise
        y0 = self.pack(u0)
        if callback:
            if callback is True:
                callback = self.callback
            callback = partial(callback, plot=plot)
        else:
            callback = None
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
        res = self._ifft(
            self._fft(self.u_noise) / (self._K2[mode] / self.lam + 1))
        if np.isrealobj(self.u_noise):
            assert np.allclose(res.imag, 0)
            res = res.real
        return res


class L1TVMaxFlow(Base):
    """Compute the L1TV denoising of an image using the PyMaxFlow library.

    Attributes
    ==========
    u_noise : array-like
        Image to denoise.
    """

    u_noise = None
    laminv2 = None
    mode = "constant"

    def __init__(self, u_noise, **kw):
        super().__init__(u_noise=u_noise, **kw)

    def init(self):
        u_noise = np.asarray(self.u_noise)
        g = maxflow.Graph[float]()
        self._nodeids = g.add_grid_nodes(u_noise.shape)
        a, b, c = 0.1221, 0.0476, 0.0454
        structure = np.array([
            [0, c, 0, c, 0],
            [c, b, a, b, c],
            [0, a, 0, a, 0],
            [c, b, a, b, c],
            [0, c, 0, c, 0],
        ])
        args = dict(symmetric=False, weights=2, structure=structure)
        if self.mode in {"constant"}:
            args.update(periodic=False)
        elif self.mode in {"wrap", "periodic"}:
            args.update(periodic=True)
        else:
            raise NotImplementedError(f"Mode {self.mode=} not implemented")
        g.add_grid_edges(self._nodeids, **args)
        self._graph = g

    def denoise1(self, threshold=0.5, laminv2=None):
        """Return the L1TV denoised image (PyMaxFlow) at a single threshold.

        Arguments
        ---------
        laminv2 : float
            Smoothing parameter 2/λ.  This defines the minimum radius of curvature of
            the level sets.
        threshold : float
            Threshold level.  Level sets for u_noise >= threshold will be computed.

        Returns
        -------
        u : array_like
            Denoised b/w image (0, 1) at the specified threshold.
        """
        if laminv2 is None:
            laminv2 = self.laminv2
        if laminv2 is None:
            raise ValueError(f"Must provide laminv2: got {laminv2=}")

        lam = 2 / laminv2
        g = self._graph.copy()
        sources = self.u_noise >= threshold
        g.add_grid_tedges(self._nodeids, lam * (1 - sources), lam * sources)
        g.maxflow()
        u = g.get_grid_segments(self._nodeids)
        return u

    def denoise(self, N=20, laminv2=None, thresholds=None, percentile=False):
        """Return the L1TV denoised image (PyMaxFlow) .

        Arguments
        ---------
        laminv2 : float
            Smoothing parameter 2/λ.  This defines the minimum radius of curvature of
            the level sets.
        N : int
            Number of thresholds to use.  Image will contain equally spaced thresholds
            between 0 and u_noise.max().
        thresholds : list, None
            List of thresholds.  These will be used instead of N equally-spaced
            thresholds if provided.
        percentile : bool
            If True, then use equally spaced percentiles for thresholds, otherwise, use
            equally spaced intensities.

        Returns
        -------
        u : array_like
            Denoised b/w image with N levels at the specified thresholds.
        """
        u = self.u_noise
        if thresholds is None:
            if percentile:
                thresholds = np.percentile(u, np.linspace(0, 100, N + 1)[1:])
            else:
                thresholds = np.linspace(u.min(), u.max(), N + 1)[1:]
        else:
            thresholds = np.sort(thresholds)
        weights = [thresholds.min()] + (np.diff(thresholds)).tolist()
        us = [
            self.denoise1(threshold=_th, laminv2=laminv2) for _th in thresholds
        ]
        return sum(_u * _w for _u, _w in zip(us, weights))


def compute_l1tv(u_noise, laminv2, threshold=0.5, mode="constant"):
    """Return the L1TV denoising of an image using the PyMaxFlow library."""
    u_noise = np.asarray(u_noise)
    lam = 2 / laminv2
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(u_noise.shape)
    a, b, c = 0.1221, 0.0476, 0.0454
    structure = np.array([
        [0, c, 0, c, 0],
        [c, b, a, b, c],
        [0, a, 0, a, 0],
        [c, b, a, b, c],
        [0, c, 0, c, 0],
    ])
    args = dict(symmetric=False, weights=1, structure=structure)
    if mode in {"constant"}:
        args.update(periodic=False)
    elif mode in {"wrap", "periodic"}:
        args.update(periodic=True)
    else:
        raise NotImplementedError(f"Mode {mode=} not implemented")

    g.add_grid_edges(nodeids, **args)
    sources = u_noise >= threshold
    g.add_grid_tedges(nodeids, lam * sources, lam * (1 - sources))
    g.maxflow()
    u = g.get_grid_segments(nodeids)
    return u


def _wkey(*x):
    """Return the square of the distance to the neighbor."""
    return np.square(x).sum()


class L1TV(Base):
    """Class to compute the L1TV denoising of an image.

    Attributes
    ==========
    u_noise : array-like
        Image to denoise.
    threshold : float
        Threshold value.  Pixels with values less than this will be treated as zero.
    mode : {"reflect", "wrap", "periodic", "constant"}
        Boundary conditions.  Same as in :func:`scipy.ndimage.convolve1d`.
    """

    u_noise = None
    threshold = 0.5
    mode = "reflect"

    _weight = {
        1: {
            _wkey(1): 1
        },
        2: {
            _wkey(0, 1): 0.1221,
            _wkey(1, 1): 0.0476,
            _wkey(1, 2): 0.0454,
        },
    }

    def __init__(self, u_noise, **kw):
        super().__init__(u_noise=u_noise, **kw)

    def init(self):
        # Compute the set of weights from [Vixie:2010].
        u_noise = self.u_noise = np.asarray(self.u_noise)
        dim = len(u_noise.shape)

        # Compute the weighted adjacency graph
        if dim == 1:
            self._weights = self.compute_weights1()
        elif dim == 2:
            self._weights = self.compute_weights2()
        else:
            raise NotImplementedError(
                f"Only 1D and 2D images supported. Got {u_noise.shape=}")

        self._connections = self.compute_connections(self.u_noise,
                                                     self.threshold)

    def compute_weights1(self):
        """Return the 1D weighted adjacency graph without the source and target.

        Returns
        -------
        weights : csr_array
            Compressed sparse row array with the weighted adjacency matrix.  This
            includes the weighted connections between the neighboring nodes as specified
            in self._weights, and includes two extra rows and columns for the source and
            target nodes, but does not include the connections with the source and target.
            See :class:`scipy.sparse.csr_array`.
        """
        dim = 1
        N = self.d.shape[0]
        weights = {}
        for n, d in itertools.product(range(N), [-1, 1]):
            t = n + d
            if self.mode == "periodic":
                t = t % N
            elif self.mode == "constant":
                if t < 0 or t >= N:
                    continue
            elif self.mode == "reflect":
                t = abs(t)
                if t >= N:
                    t = 2 * N - t - 1
            wkey = _wkey(d)
            key = (n, t)
            keyT = (t, n)
            weight_dict = self._weight[dim]
            if wkey in weight_dict and key not in weights and keyT not in weights:
                weights[key] = weights[keyT] = weight_dict[wkey]

        rows = [_k[0] for _k in weights]
        cols = [_k[1] for _k in weights]
        vals = list(weights.values())

        return sp.sparse.csr_matrix((vals, (rows, cols)), shape=(N + 2, ) * 2)

    def compute_weights2(self):
        """Return the 2D weighted adjacency graph without the source and target.

        Returns
        -------
        weights : csr_array
            Compressed sparse row array with the weighted adjacency matrix.  This
            includes the weighted connections between the neighboring nodes as specified
            in self._weights, and includes two extra rows and columns for the source and
            target nodes, but does not include the connections with the source and target.
            See :class:`scipy.sparse.csr_array`.
        """
        dim = 2
        Nx, Ny = self.u_noise.shape
        weights = {}
        for nx, ny, dx, dy in itertools.product(range(Nx), range(Ny),
                                                *([-2, -1, 0, 2, 1], ) * 2):
            tx, ty = nx + dx, ny + dy
            if self.mode == "periodic":
                tx, ty = tx % Nx, ty % Ny
            elif self.mode == "constant":
                if tx < 0 or ty < 0 or tx >= Nx or ty >= Ny:
                    continue
            elif self.mode == "reflect":
                tx, ty = abs(tx), abs(ty)
                if tx >= Nx:
                    tx = 2 * Nx - tx - 1
                if ty >= Ny:
                    ty = 2 * Ny - ty - 1
            wkey = _wkey(dx, dy)
            key = ((nx, ny), (tx, ty))
            keyT = key[::-1]
            weight_dict = self._weight[dim]
            if wkey in weight_dict and key not in weights and keyT not in weights:
                weights[key] = weights[keyT] = weight_dict[wkey]

        def ind(nx, ny):
            return ny + nx * Ny

        rows = [ind(*_k[0]) for _k in weights]
        cols = [ind(*_k[1]) for _k in weights]
        vals = list(weights.values())

        return sp.sparse.csr_matrix((vals, (rows, cols)),
                                    shape=(Nx * Ny + 2, ) * 2)

    def compute_connections(self, u, threshold=1):
        """Return the adjacency graph connecting the source and target.

        Returns
        -------
        connections : csr_array
            Compressed sparse row array with the adjacency matrix.  This includes
            connections between the source with the pixels >= threshold, and the target
            with the pixels < threshold.  Does not have the factor λ. See
           :class:`scipy.sparse.csr_array`.
        """
        u = np.ravel(u)
        s = len(u)
        t = s + 1
        t_rows = np.where(u < threshold)[0]
        s_cols = np.where(u >= threshold)[0]
        s_rows = s * np.ones(len(s_cols), dtype=int)
        t_cols = t * np.ones(len(t_rows), dtype=int)

        rows = np.concatenate([s_rows, t_rows])
        cols = np.concatenate([s_cols, t_cols])
        vals = np.ones(len(rows))
        C = sp.sparse.csr_matrix((vals, (rows, cols)),
                                 shape=(len(u) + 2, ) * 2)
        return C + C.T

    def denoise(self, laminv2):
        """Return u, the denoised image.

        Parameters
        ----------
        laminv2 : float
            Smoothing parameter 2/λ.  This represents the minimum radius of curvature of
            the level sets in the smoothed image (in pixels).
        """
        lam = 2 / laminv2
        N = np.prod(self.u_noise.shape)
        s = N
        t = N + 1
        import flow

        f = flow.Flow(self._weights + self._connections * lam, s=s, t=t)
        cut = f.min_cut()
        S, T = cut.S, cut.T
        S.remove(f.s)
        T.remove(f.t)
        u = np.zeros_like(self.u_noise, dtype=int)
        u_ = u.view()
        u_.shape = N  # Flat view
        u_[T] = 1
        return u


class NonLocalMeans(Base):
    """Non-local means denoising.

    Attributes
    ----------
    image : Image
        Instance of :class:`Image` with the image data.
    dx, dy : int
        Width and height of the patches (in pixels).
    sigma : float
        Standard deviation (as a fraction) of gaussian noise to add to the image.
    symmetric : bool
        True if the distance function is symmetric.
    subtract_mean : bool
        If True, then subtract the patch mean in the differences.
    """

    image = None
    dx = 5
    dy = 5
    mode = "wrap"

    sigma = 0.5
    seed = 2
    symmetric = True
    subtract_mean = True

    def __init__(self, image=None, **kw):
        super().__init__(image=image, **kw)

    def init(self):
        self.rng = np.random.default_rng(seed=self.seed)
        if self.image is not None:
            self.u_exact = self.image.get_data(sigma=0, normalize=True)
            self.u_noise = self.image.get_data(sigma=self.sigma,
                                               normalize=True,
                                               rng=self.rng)

    def pad(self, u=None):
        """Return the padded array.  This implements the boundaries."""
        if u is None:
            u = self.u_noise
        Nx, Ny = u.shape
        dx, dy = self.dx, self.dy

        # These likely fail if the following conditions are not met
        assert dx // 2 <= Nx
        assert dy // 2 <= Ny
        Nx_, Ny_ = Nx + dx - 1, Ny + dy - 1
        u_ = np.zeros_like(u, shape=(Nx_, Ny_))
        ix0, iy0 = dx // 2, dy // 2
        ix1, iy1 = (dx - 1) // 2, (dy - 1) // 2
        u_[ix0:ix0 + Nx, iy0:iy0 + Ny] = u
        if self.mode == "constant":
            return u_
        if self.mode in {"wrap", "periodic"}:
            u_[:ix0, :] = u_[Nx:, :][:ix0, :]
            u_[-ix1:, :] = u_[ix0:, :][:ix1, :]
            u_[:, :iy0] = u_[:, Ny:][:, :iy0]
            u_[:, -iy1:] = u_[:, iy0:][:, :iy1]
            return u_
        if self.mode == "nearest":
            u_[:ix0, :] = u_[ix0, :][None, :]
            u_[-ix1:, :] = u_[Nx + 1, :][None, :]
            u_[:, :iy0] = u_[:, iy0][:, None]
            u_[:, -iy1:] = u_[:, Ny + 1][:, None]
            return u_
        if self.mode == "reflect":
            u_[:ix0, :] = u_[ix0:, :][:ix0, :][::-1, :]
            u_[-ix1:, :] = u_[Nx + 1::-1, :][:ix1, :]
            u_[:, :iy0] = u_[:, iy0:][:, :iy0][:, ::-1]
            u_[:, -iy1:] = u_[:, Ny + 1::-1][:, :iy1]
            return u_
        if self.mode == "mirror":
            u_[:ix0, :] = u_[ix0 + 1:, :][:ix0, :][::-1, :]
            u_[-ix1:, :] = u_[Nx::-1, :][:ix1, :]
            u_[:, :iy0] = u_[:, iy0 + 1:][:, :iy0][:, ::-1]
            u_[:, -iy1:] = u_[:, Ny::-1][:, :iy1]
            u_[:ix0, :] = u_[ix0 + 1:, :][:ix0, :][::-1, :]
            return u_
        raise ValueError(f"Unsupported {self.mode=}")

    def dist(self, A, B, _internal=False):
        """Return the d, or (d, dAB_), the distance between patches A and B and the mean.

        _internal : bool
            If True, then use self.subtract_mean and return dAB_.  Otherwise do not
            subtract the mean and return only d.  The latter is useful for users when
            they want to compute the distance between images.

        Returns
        -------
        d : float
            The distance between A and B.
        dAB_ : float
            (A-B).mean().  If self.subtract_mean, then this is subtracted from d, but
            must be added back when reconstructing the image.  If not
            self.subtract_mean, then dAB_ is zero.  Only returned if _internal == True.
        """
        dAB = A - B
        dAB_ = 0
        if _internal and self.subtract_mean:
            dAB_ = dAB.mean()
            dAB -= dAB_

        d = (dAB**2).mean()
        if _internal:
            return (d, dAB_)
        return d

    def ixy(self, i, Nx=None):
        """Return the pixel index (ix, iy) of patch i.

        Note: ix goes down and iy goes across (indexing='ij') so that
        `(Nx, Ny) = self.u_noise.shape`.  This is the transpose of plt.imshow() where ix
        goes across and iy goes down..
        """
        if Nx is None:
            Nx = self.u_noise.shape[0]
        ix = i % Nx
        iy = i // Nx
        return (ix, iy)

    def i(self, ix, iy, Nx=None):
        """Return the index i for the patch at pixel (ix, iy).

        Note: ix goes down and iy goes across (indexing='ij') so that
        `(Nx, Ny) = self.u_noise.shape`.  This is the transpose of plt.imshow() where ix
        goes across and iy goes down..
        """
        if Nx is None:
            Nx = self.u_noise.shape[0]
        return ix + Nx * iy

    def get_threshold(self, percentile=96, Nsamples=None, rng=None, seed=2):
        """Return the threshold distance.

        The idea here is that, if two patches only deviate because of the noise, then we
        should be able to characterize the distribution of the distances, and we want to
        capture at least the specified percentile of the data.

        Arguments
        ---------
        percentile : float
            The threshold will be at this percentile.  Thus, if this is 96, then 96% of
            the randomly produced samples will be captured.
        Nsamples : int
            If None, then assume that `self.dist()` is the canonical example which is
            the mean of the square of the deviations.  The resulting distribution is a
            chi-squared distribution with `dx*dy` degrees of freedom.  Otherwise, we
            will numerically deduce the distribution from Nsamples independent samples.
        rng, seed :
            User can provide a seed, or a random number generator if needed.
        """
        if Nsamples is None:
            N = df = self.dx * self.dy
            if self.subtract_mean:
                df -= 1
            chi2 = sp.stats.chi2(scale=2 * self.sigma**2 / N, df=df)
            threshold = chi2.ppf(percentile / 100)
            return threshold

        if rng is None:
            rng = np.random.default_rng(seed=seed)

        size = (2, self.dx, self.dy)
        threshold = np.percentile(
            [
                self.dist(*rng.normal(scale=self.sigma, size=size),
                          _internal=True)[0] for _N in range(Nsamples)
            ],
            percentile,
        )
        return threshold

    def get_patch(self, i, u_=None):
        """Return the ith patch."""
        if u_ is None:
            u_ = self.pad(self.u_noise)
        Nx = u_.shape[0] - (self.dx - 1)
        ix, iy = self.ixy(i, Nx=Nx)
        return u_[ix:ix + self.dx, iy:iy + self.dy]

    def compute_dists(self, u=None, u_=None):
        """Return the array of (dist, dAB_) pairs: the distances between the patches.

        Returns
        -------
        dist : array-like
            Array of (dist, dAB_) entries where dist is the distance between the
            patches, and dAB_ is the mean of the patch difference.  The array thus has
            shape (Nx*Ny, Nx*Ny, 2)
        """
        dx, dy = self.dx, self.dy
        if u_ is None:
            if u is None:
                u = self.u_noise
            u_ = self.pad(u)
            Nx, Ny = u.shape
        else:
            Nx, Ny = np.subtract(u_.shape, (dx - 1, dy - 1))

        Np = Nx * Ny
        if self.symmetric:
            dists = np.array([[[0, 0]] * (i1 + 1) + [
                self.dist(
                    self.get_patch(i0, u_=u_),
                    self.get_patch(i1, u_=u_),
                    _internal=True,
                ) for i0 in range(i1 + 1, Np)
            ] for i1 in range(Np)])
            # Transpose only on the first two indices.  Note that dAB_ changes sign for
            # the transposed entries.
            dists += np.einsum("ab...->ba...", dists) * np.array([1, -1])[:,
                                                                          ...]
        else:
            dists = np.array([[
                self.dist(
                    self.get_patch(i0, u_=u_),
                    self.get_patch(i1, u_=u_),
                    _internal=True,
                ) for i0 in range(Np)
            ] for i1 in range(Np)])
        return dists

    def denoise(
        self,
        u=None,
        percentile=96,
        Nmin=0,
        f_weight=None,
        sigma_wight=None,
        k_sigma=1.0,
        dists=None,
        u_=None,
        u_exact=None,
        debug=False,
    ):
        """Denoise the image u.

        Parameters
        ----------
        percentile : float, 'optimize'
            Percentile to use for determining the threshold.  See
            {func}`get_threshold`.  If set to `optimize` and `u_exact` is provided, then
            then :func:`scipy.optimize.minimize_scalar` will be used to optimize the
            value.
        f_weight : function, None
            Function f_weight(d, sigma_weight) that returns the relative weight of a pixel
            distance d away.  Default is a gaussian with width sigma_weight.
        sigma_weight : function, None
            Function sigma_weight(ds) that returns the sigma_weight passed to f_weight
            given the neighborhood of distances ds.  The default is max(ds) / 2 / k_sigma.
        k_sigma : float
            Parameter in the default sigma_weight.  The diameter of the neighbourhood
            ``max(ds) = 2*k_sigma * sigma_weight``.  Set to zero for constant weights
            over the neighbourhood.
        dists, u_ :
            If provided, used to improve performance.
        debug : bool
            If True, then return a named tuple with intermediate information like the
            dists for analysis.
        """
        if u is None:
            u = self.u_noise
        if u_ is None:
            u_ = self.pad(u)
        if dists is None:
            dists = self.compute_dists(u_=u_)

        if percentile == "optimize":
            if u_exact is None:
                u_exact = self.u_exact

            _cache = {}

            def err(percentile):
                if percentile not in _cache:
                    _cache[percentile] = self.dist(
                        u_exact,
                        self.denoise(u=u,
                                     dists=dists,
                                     u_=u_,
                                     percentile=percentile),
                    )
                return _cache[percentile]

            res = sp.optimize.minimize_scalar(err,
                                              bracket=(10, 50),
                                              bounds=(0, 100),
                                              options=dict(xatol=0.1))
            if not res.success:
                raise ValueError(res.message)

            percentile = res.x
            err = res.fun
            print(f"Optimal {percentile=:.1f}: {err=:.2g}")

        self._percentile = percentile

        Nx, Ny = u.shape
        Np = Nx * Ny

        if self.symmetric:
            ds = sorted([(dists[i0, i1].tolist(), (i0, i1)) for i1 in range(Np)
                         for i0 in range(i1 + 1, Np)])
        else:
            ds = sorted([(dists[i0, i1].tolist(), (i0, i1)) for i1 in range(Np)
                         for i0 in range(Np) if i0 != i1])

        # Build Graph: keys are nodes, values are lists of neighbours
        if f_weight is None:

            def sigma_weight(ds):
                """Return the weight give the neighbourhood distance ds."""
                return max(ds) / 2 / max(k_sigma, _EPS)

            def f_weight(d, sigma):
                """Return the weight given distance d.

                Arguments
                ---------
                d : float
                    Distance to the current point.
                sigma : float
                    Reference distance returned by sigma_weight()
                """
                return np.exp(-((d / sigma)**2) / 2)

        G = {}
        threshold = self.get_threshold(percentile=percentile)
        for ((d, dAB_), (i0, i1)) in ds:
            neighbours = G.setdefault(i0, [])

            if d <= threshold or len(neighbours) < Nmin:
                neighbours.append((i1, (d, dAB_)))
            if self.symmetric:
                neighbours = G.setdefault(i1, [])
                if d <= threshold or len(neighbours) < Nmin:
                    # Remember that the transposed entries have the opposite sign of dAB_
                    neighbours.append((i0, (d, -dAB_)))

        # Actually denoise the image.
        u_clean = []
        for i0 in range(Np):
            ix, iy = self.ixy(i0, Nx=Nx)
            us = [u[ix, iy]]
            ws = [f_weight(0, sigma=1)]
            d_dABs = G.get(i0, [])
            if d_dABs:
                ds = [_d for (_d, _d_) in d_dABs]
                sigma = sigma_weight(ds)
                for i1, (d1, dAB_) in d_dABs:
                    i1x, i1y = self.ixy(i1, Nx=Nx)
                    us.append(u[i1x, i1y] - dAB_)
                    ws.append(f_weight(d1, sigma=sigma))

            u_clean.append(np.dot(us, ws) / sum(ws))
        u_clean = np.array(u_clean).reshape((Ny, Nx)).T
        if not debug:
            return u_clean
        DebugResults = namedtuple("DebugResults",
                                  ["u_clean", "u_", "dists", "G"])
        return DebugResults(u_clean=u_clean, u_=u_, dists=dists, G=G)


class CharacteristicGraphs(Base):
    """

    Attributes
    ----------
    u_noise : array-like
        Image to denoise.
    kind : {'G0', 'K1', 'K2'}
        Type of connectivity (see [Asaki:2010]).
    """

    u = None
    kind = "K1"

    def __init__(self, u, **kw):
        super().__init__(u=u, **kw)

    def init(self):
        # Indices
        u = self.u
        Nx, Ny = u.shape
        ix, iy = np.mgrid[:Nx, :Ny]
        wx = abs(np.diff(u, axis=0))  # vertical edge weights
        wy = abs(np.diff(u, axis=1))  # horizontal edge weights

        self.V0 = set(list(zip(ix.ravel(), iy.ravel())))  # All vertices

        # Get all edges in G0 and corresponding weights.
        E0 = [
            (_ixy, 0) for _ixy in zip(ix[:-1, :].ravel(), iy[:-1, :].ravel())
        ] + [(_ixy, 1) for _ixy in zip(ix[:, :-1].ravel(), iy[:, :-1].ravel())]
        w0 = np.concatenate([wx.ravel(), wy.ravel()])

        self.E0 = set(E0)
        ## Sort by weight
        self._w0, self._E0 = zip(*sorted(zip(w0, E0)))
        self.weights = dict(zip(E0, w0))

    def compute_G0(self):
        """Return (V, E)=(V0, E0) defining the G0 connectivity graph."""
        return (self.V0, self.E0)

    def compute_K1(self):
        """Return (V, E) defining the K1 connectivity graph."""
        # 1. Begin with an empty graph V ={} and E={}.
        clusters = []
        V = set()
        E = set()

        # 2. Add to E the edge of minimum weight in E0 that does not create a cycle in (V, E).
        # Add the corresponding vertices to V that are not already in V .
        # 3. Repeat step 2 until V = V0.
        edges = tqdm(self._E0)
        for e in edges:
            va, vb = self.get_vertices(e)
            na = nb = None
            assert e not in E
            for _n, (_V, _E) in enumerate(clusters):
                if va in _V:
                    na = _n
                if vb in _V:
                    nb = _n
            if na is None and nb is None:
                # New cluster
                clusters.append(({va, vb}, {e}))
                assert va not in V and vb not in V
            elif na is None or nb is None:
                # Grow a cluster
                if na is None:
                    _V, _E = clusters[nb]
                    _V.add(va)
                    _E.add(e)
                    assert va not in V and vb in V
                else:
                    _V, _E = clusters[na]
                    _V.add(vb)
                    _E.add(e)
                    assert va in V and vb not in V
            elif na != nb:
                # Connect two clusters
                _Va, _Ea = clusters[na]
                _Vb, _Eb = clusters.pop(nb)
                _Va.update(_Vb)
                _Ea.update(_Eb)
                _Ea.add(e)
                assert va in V and vb in V
            else:
                # Edge would create a cycle
                continue
            V.update({va, vb})
            E.add(e)
            edges.set_description(
                f"{len(V)} of {len(self.V0)}: {len(clusters)} clusters")
            if V == self.V0:
                break

        # 4. Find the edge of minimum weight in E0 that does not create a cycle in (V, E).
        # Set wcut equal to the corresponding edge weight.
        edges = tqdm([e for e in self._E0 if e not in E])
        for n, e in enumerate(edges):
            va, vb = self.get_vertices(e)
            for _n, (_V, _E) in enumerate(clusters):
                if va in _V:
                    na = _n
                if vb in _V:
                    nb = _n
            assert na is not None and nb is not None
            if na == nb:
                # Edge would form a cycle
                continue
            break

        wcut = self.weights[e]

        # Compute degrees
        degs = {_v: 0 for _v in V}
        for e in E:
            va, vb = self.get_vertices(e)
            degs[va] += 1
            degs[vb] += 1

        # 5. To each vertex of degree 1 add to E the associated edge from E0 that
        #    (a) has smallest weight,
        #    (b) is not already in E, and
        #    (c) if the edge weight is less than wcut.
        deg1 = [v for v in degs if degs[v] == 1]
        for v in deg1:
            for _w, e in sorted(
                (self.weights[_e], _e) for _e in self.get_edges(v)):
                if _w < wcut and e not in E:
                    E.add(e)
                    for _V, _E in clusters:
                        if v in _V:
                            _E.add(e)
                    break
        return (V, E)

    def compute_K2(self):
        """Return (V, E) defining the K2 vertex inclusion graph."""
        # 1. Begin with an empty graph V ={} and E={}.
        V = set()
        E = set()

        # 2. Add the edge of smallest weight from E0 to E that is not already in E, and
        #    the corresponding vertices from V0 to V that are not already in V .
        # 3. Repeat step 2 until V = V0.
        edges = tqdm(self._E0)
        for e in edges:
            va, vb = self.get_vertices(e)
            V.update({va, vb})
            E.add(e)
            if V == self.V0:
                break
            edges.set_description(f"{len(V)} of {len(self.V0)}")

        # 4. Add all edges from E0 to E that are of equal weight to the largest edge weight in E.
        wmax = max(self.weights[_e] for _e in E)
        for e in [e for e in self.E0.difference(E) if self.weights[e] == wmax]:
            V.update(self.get_vertices(e))
            E.add(e)
        return (V, E)

    @staticmethod
    def get_vertices(edge):
        """Return the vertices connected by the edge."""
        va, d = edge
        vb = list(va)
        vb[d] += 1
        vb = tuple(vb)
        return (va, vb)

    def get_edges(self, v):
        """Return the adjacent edges."""
        ix, iy = v
        Nx, Ny = self.u.shape
        edges = []
        if ix > 0:
            edges.append(((ix - 1, iy), 0))
        if iy > 0:
            edges.append(((ix, iy - 1), 1))
        if ix < Nx - 1:
            edges.append(((ix, iy), 0))
        if iy < Ny - 1:
            edges.append(((ix, iy), 1))
        return edges

    def draw_graph(self, E, weights=None, ax=None, **kw):
        if ax is None:
            fig, ax = subplots()
        segments = []
        for e in E:
            va, vb = self.get_vertices(e)
            segments.append((va, vb))
        if weights is not None:
            kw["colors"] = cm.viridis(weights)
        ax.add_collection(LineCollection(segments, **kw))
        ax.autoscale()
        ax.set(aspect=1)
