"""Element metadata: atomic-number ordering and periodic-table colours.

Shared by the composition/detection reporting (``alibz.detections``,
``alibz.pipeline``) and the inspection notebooks so every chart orders
and colours elements the same way.  Pure data + pure functions, no heavy
imports — safe to pull in anywhere.
"""

import colorsys
from typing import Dict, Tuple

ELEMENTS_BY_ATOMIC_NUMBER = (
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
)
ATOMIC_NUMBER = {el: i + 1 for i, el in enumerate(ELEMENTS_BY_ATOMIC_NUMBER)}

PERIODIC_BLOCK_MEMBERS = {
    "reactive nonmetal": ("H", "C", "N", "O", "P", "S", "Se"),
    "group 1": ("Li", "Na", "K", "Rb", "Cs", "Fr"),
    "group 2": ("Be", "Mg", "Ca", "Sr", "Ba", "Ra"),
    "3d-block": ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
                 "Cu", "Zn"),
    "4d-block": ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                 "Ag", "Cd"),
    "5d-block": ("Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
                 "Hg"),
    "4f-block": ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                 "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"),
    "5f-block": ("Ac", "Th", "Pa", "U"),
    "post-transition metal": ("Al", "Ga", "In", "Sn", "Tl", "Pb",
                              "Bi"),
    "metalloid": ("B", "Si", "Ge", "As", "Sb", "Te", "Po"),
    "halogen": ("F", "Cl", "Br", "I", "At"),
    "noble gas": ("He", "Ne", "Ar", "Kr", "Xe", "Rn"),
}
ELEMENT_PERIODIC_BLOCK = {
    el: block
    for block, elements in PERIODIC_BLOCK_MEMBERS.items()
    for el in elements
}
PERIODIC_BLOCK_COLORS = {
    "reactive nonmetal": "#dc2626",
    "group 1": "#e11d48",
    "group 2": "#d97706",
    "3d-block": "#2563eb",
    "4d-block": "#0d9488",
    "5d-block": "#7c3aed",
    "4f-block": "#c026d3",
    "5f-block": "#92400e",
    "post-transition metal": "#64748b",
    "metalloid": "#65a30d",
    "halogen": "#16a34a",
    "noble gas": "#0891b2",
    "other": "#52525b",
}
PERIODIC_BLOCK_COLOR_STYLE = {
    # hue degrees, saturation, light shade, dark shade
    "reactive nonmetal": (4.0, 0.76, 0.82, 0.38),
    "group 1": (342.0, 0.78, 0.84, 0.38),
    "group 2": (36.0, 0.82, 0.84, 0.36),
    "3d-block": (215.0, 0.78, 0.82, 0.34),
    "4d-block": (174.0, 0.74, 0.80, 0.32),
    "5d-block": (262.0, 0.72, 0.82, 0.36),
    "4f-block": (292.0, 0.68, 0.84, 0.38),
    "5f-block": (22.0, 0.72, 0.82, 0.35),
    "post-transition metal": (210.0, 0.34, 0.78, 0.36),
    "metalloid": (78.0, 0.66, 0.78, 0.34),
    "halogen": (132.0, 0.68, 0.78, 0.34),
    "noble gas": (190.0, 0.68, 0.78, 0.34),
    "other": (0.0, 0.0, 0.60, 0.40),
}


def _hex_to_rgb01(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    vals = [max(0, min(255, int(round(v * 255.0)))) for v in rgb]
    return "#{:02x}{:02x}{:02x}".format(*vals)


def _build_element_colors() -> Dict[str, str]:
    """Unique shades grouped by periodic-table chemistry/position."""
    colors = {}
    for block, members in PERIODIC_BLOCK_MEMBERS.items():
        hue_deg, saturation, light_hi, light_lo = PERIODIC_BLOCK_COLOR_STYLE[
            block
        ]
        ordered = sorted(members, key=lambda el: (ATOMIC_NUMBER[el], el))
        n = len(ordered)
        for i, el in enumerate(ordered):
            frac = 0.5 if n == 1 else i / (n - 1)
            # A small within-family hue drift plus a wide lightness span
            # keeps related elements grouped without making adjacent
            # members look like repeated colors in stacked bars.
            hue = ((hue_deg + (frac - 0.5) * 28.0) % 360.0) / 360.0
            lightness = light_hi - (light_hi - light_lo) * frac
            sat = max(0.20, min(0.90, saturation + 0.06 * (0.5 - frac)))
            colors[el] = _rgb01_to_hex(colorsys.hls_to_rgb(
                hue, lightness, sat
            ))
    return colors


ELEMENT_COLORS = _build_element_colors()


def element_sort_key(element: str) -> Tuple[int, str]:
    """Sort key for periodic-table order, with unknown labels last."""
    return ATOMIC_NUMBER.get(element, 10_000), element


def element_periodic_block(element: str) -> str:
    """Periodic-table block/family used for inspection-notebook coloring."""
    return ELEMENT_PERIODIC_BLOCK.get(element, "other")


def element_block_color(element: str) -> str:
    """Unique same-hue shade assigned to an element's periodic block."""
    return PERIODIC_BLOCK_COLORS[element_periodic_block(element)]


def element_color(element: str) -> str:
    """Unique same-hue shade assigned to an individual element."""
    return ELEMENT_COLORS.get(element, PERIODIC_BLOCK_COLORS["other"])
