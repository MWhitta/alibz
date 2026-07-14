"""Map observed wavelengths onto their perceived visible-spectrum colour.

A LIBS spectrum spans ~180-950 nm — from the ultraviolet, through the
visible band, into the near-infrared.  Colouring spectral features by the
colour the eye WOULD see at that wavelength makes plots read at a glance: a
Li line at 670 nm is red, a Ca II line at 393 nm is violet, an Fe II line at
259 nm is (dim) ultraviolet.  This module provides one canonical mapping so
every plot in the repo shares it.

``wavelength_to_rgb`` uses the standard visible-spectrum colour ramp (after
Dan Bruton) over 380-750 nm; outside that the hue is clamped to the violet
(UV) or red (NIR) end and dimmed toward a floor, so invisible-band features
stay on-screen but clearly read as "beyond violet" / "beyond red".
"""
from typing import Optional

import numpy as np

#: default visible-spectrum span used for the colour ramp [nm]
VISIBLE_LO, VISIBLE_HI = 380.0, 750.0
#: default full LIBS span used for colormaps / gradient backgrounds [nm]
LIBS_LO, LIBS_HI = 180.0, 950.0


def wavelength_to_rgb(wavelength, gamma: float = 0.85,
                      floor: float = 0.22) -> np.ndarray:
    """Perceived RGB colour(s) of ``wavelength`` [nm], in [0, 1].

    Vectorised: a scalar returns shape ``(3,)``; an array of shape ``S``
    returns ``S + (3,)``.  ``floor`` is the residual brightness kept in the
    UV/NIR so those features remain visible; ``gamma`` shapes the ramp.
    """
    w = np.asarray(wavelength, dtype=float)
    scalar = (w.ndim == 0)
    w = np.atleast_1d(w).astype(float)
    wc = np.clip(w, VISIBLE_LO, VISIBLE_HI)   # hue clamps at the band edges

    r = np.zeros_like(wc)
    g = np.zeros_like(wc)
    b = np.zeros_like(wc)

    def band(lo, hi, rf, gf, bf):
        m = (wc >= lo) & (wc <= hi)
        if not np.any(m):
            return
        t = (wc[m] - lo) / (hi - lo)
        r[m], g[m], b[m] = rf(t), gf(t), bf(t)

    one = np.ones_like
    zero = np.zeros_like
    band(380, 440, lambda t: 1.0 - t, zero, one)          # violet -> blue
    band(440, 490, zero, lambda t: t, one)                # blue -> cyan
    band(490, 510, zero, one, lambda t: 1.0 - t)          # cyan -> green
    band(510, 580, lambda t: t, one, zero)                # green -> yellow
    band(580, 645, one, lambda t: 1.0 - t, zero)          # yellow -> red
    band(645, 750, one, zero, zero)                       # red

    # brightness envelope: full through the visible core, dimming into the
    # UV (below ~420) and NIR (above ~700), never darker than ``floor``.
    factor = np.ones_like(w)
    lo_vis = (w < 420) & (w >= VISIBLE_LO)
    factor[lo_vis] = 0.30 + 0.70 * (w[lo_vis] - VISIBLE_LO) / 40.0
    hi_vis = (w > 700) & (w <= VISIBLE_HI)
    factor[hi_vis] = 0.30 + 0.70 * (VISIBLE_HI - w[hi_vis]) / 50.0
    uv = w < VISIBLE_LO
    factor[uv] = np.interp(w[uv], [LIBS_LO, VISIBLE_LO], [floor, 0.30])
    nir = w > VISIBLE_HI
    factor[nir] = np.interp(w[nir], [VISIBLE_HI, 1000.0], [0.30, floor])
    factor = np.clip(factor, floor, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0) ** gamma
    rgb = rgb * factor[..., None]
    return rgb[0] if scalar else rgb


def spectral_colormap(lo: float = LIBS_LO, hi: float = LIBS_HI,
                      n: int = 512):
    """A matplotlib colormap of the wavelength colour ramp over ``[lo, hi]``.

    Use with ``c=wavelengths, cmap=spectral_colormap(), vmin=lo, vmax=hi``
    to colour scatter/line collections by wavelength.
    """
    from matplotlib.colors import ListedColormap
    grid = np.linspace(lo, hi, int(n))
    return ListedColormap(wavelength_to_rgb(grid), name="libs_spectral")


def spectral_background(ax, lo: Optional[float] = None,
                        hi: Optional[float] = None, alpha: float = 0.13,
                        zorder: float = -10) -> None:
    """Paint a faint wavelength-colour gradient behind a spectrum on ``ax``.

    The x-axis is assumed to be wavelength [nm]; ``lo``/``hi`` default to the
    current x-limits.  Drawn under everything (``zorder``) at low ``alpha``
    so it tints the background without competing with the data.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo = x0 if lo is None else lo
    hi = x1 if hi is None else hi
    grid = np.linspace(lo, hi, 512)
    strip = wavelength_to_rgb(grid)[None, :, :]
    ax.imshow(strip, extent=(lo, hi, 0, 1), aspect="auto",
              transform=ax.get_xaxis_transform(), origin="lower",
              alpha=alpha, zorder=zorder, interpolation="bilinear")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def spectral_line(ax, x, y, lw: float = 0.9, alpha: float = 1.0,
                  zorder: float = 2, label=None, **kwargs):
    """Plot ``y`` vs wavelength ``x`` as a line coloured BY wavelength.

    Each segment takes the colour the eye would see at its wavelength, so
    the trace itself carries the spectral scheme — with no background
    artist, the intensity (y) axis is left completely untouched (linear or
    log).  Autoscaling and axis limits behave exactly as a normal
    ``ax.plot`` because an invisible companion line drives them; the visible
    colour comes from a ``LineCollection`` on top.  Returns the collection.
    """
    from matplotlib.collections import LineCollection

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # invisible companion drives autoscale / limits / log handling and
    # carries the legend entry, so this is a drop-in for ax.plot
    ax.plot(x, y, alpha=0.0, zorder=zorder, label=label)
    if x.size < 2:
        return None
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    mid = 0.5 * (x[:-1] + x[1:])
    lc = LineCollection(segs, colors=wavelength_to_rgb(mid), linewidths=lw,
                        alpha=alpha, zorder=zorder, **kwargs)
    # autolim=False: the collection must NOT touch the data limits (its raw
    # y-extent would drag a log intensity axis down to 0); the invisible
    # companion line is the sole autoscale driver, so limits match ax.plot
    ax.add_collection(lc, autolim=False)
    return lc
