#!/usr/bin/env python
"""Decode a SciAps Z300 ``all_fb`` spectrum bundle into per-shot CSVs.

The Z300 keeps every test's raw spectra in its internal Couchbase Lite
database (``libzdata/libzdb.cblite`` on the SD card).  Each test document
carries a ``shotTable`` whose ``all_fb`` entry names a file under
``libzdata/spectra/`` holding *all* shots of that test as a FlatBuffers
blob.  The instrument only writes the friendly
``export/geochem_pro_spectra/Test …/Shot(N).csv`` files when an operator
performs an export; this module recovers the same spectra without one.

The format was reverse-engineered from the bytes (there is no schema in the
file) and then validated against the vendor's own export -- see
``compare_with_csv``.  Layout::

    root table
      field 3 -> vector of N shot tables            (N = rasterNumLocations)

    shot table
      field 0 -> vector of 5 float64                segment edges, nm
                                                    (180, 365, 620, 960, 961)
      field 1 -> vector of 4 segment tables
      field 2 -> vector, empty in every test seen

    segment table
      field 0 -> vector of 4 float64                wavelength polynomial
      field 1 -> vector of 2066 float64             intensities, dark-subtracted

Wavelength for stored sample ``i`` of a segment is ``sum(c[k] * (i + PIXEL_OFFSET)**k)``,
a cubic in detector pixel index that runs *descending* in wavelength.  The
polynomial is defined against the physical pixel column, while the stored
array starts 18 columns later -- presumably masked/dark pixels the firmware
trims before writing.  ``PIXEL_OFFSET = -18`` was solved for, not assumed:
scanning the offset against the vendor's export maximises the correlation at
exactly -18.0 px in every segment of every shot checked, lifting agreement
from r ~ 0.05 (no offset) to r ~ 0.91-0.98.  Getting this wrong shifts lines
by up to ~3.5 nm, which silently mis-assigns elements.

The four segments overlap; the instrument stitches them by taking each
segment only within its slot of the ``field 0`` edge list, which is what
:func:`stitch` reproduces.

Segment 3 is a 1 nm stub (960 -> 960.18) and contributes almost nothing; it
is kept only so the reconstruction matches the vendor grid exactly.

Usage::

    python scripts/z300_fb_decode.py BUNDLE --out-dir DIR [--prefix rep1]
    python scripts/z300_fb_decode.py BUNDLE --compare 'Shot(1).csv' --shot 0
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
from typing import List, Optional, Sequence, Tuple

import numpy as np

# The vendor's exported CSV grid: 180.0 .. 960.9 nm inclusive, 0.1 nm steps.
EXPORT_LO_NM = 180.0
EXPORT_STEP_NM = 0.1
EXPORT_N = 7811

PIXELS_PER_SEGMENT = 2066
N_SEGMENTS = 4

# Detector columns between the wavelength polynomial's origin and the first
# stored sample.  Solved against the vendor export; see the module docstring.
PIXEL_OFFSET = -18.0


class FlatBufferError(ValueError):
    """The bundle does not have the expected Z300 FlatBuffers shape."""


class _Reader:
    """Minimal little-endian FlatBuffers accessor (no generated code)."""

    def __init__(self, buf: bytes):
        self.b = buf

    def u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.b, off)[0]

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.b, off)[0]

    def i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.b, off)[0]

    def vtable(self, table: int) -> List[int]:
        """Field offsets of ``table``; 0 means the field is absent."""
        vt = table - self.i32(table)
        if not 0 <= vt < len(self.b):
            raise FlatBufferError(f"vtable out of range at table {table}")
        vlen = self.u16(vt)
        return [self.u16(vt + 4 + 2 * i) for i in range((vlen - 4) // 2)]

    def field(self, table: int, index: int) -> Optional[int]:
        entries = self.vtable(table)
        if index >= len(entries) or entries[index] == 0:
            return None
        return table + entries[index]

    def follow(self, pos: int) -> int:
        """Resolve a uoffset stored at ``pos`` to its absolute target."""
        return pos + self.u32(pos)

    def vector(self, pos: int) -> Tuple[int, int]:
        """Return ``(length, data_offset)`` for the vector pointed at by pos."""
        target = self.follow(pos)
        return self.u32(target), target + 4

    def doubles(self, pos: int) -> np.ndarray:
        n, data = self.vector(pos)
        if data + 8 * n > len(self.b):
            raise FlatBufferError(f"double vector of {n} overruns the buffer")
        return np.frombuffer(self.b, dtype="<f8", count=n, offset=data)


def _shot_tables(reader: _Reader) -> List[int]:
    root = reader.u32(0)
    pos = reader.field(root, 3)
    if pos is None:
        raise FlatBufferError("root table has no shot vector (field 3)")
    n, data = reader.vector(pos)
    if not 0 < n < 10000:
        raise FlatBufferError(f"implausible shot count {n}")
    return [data + 4 * i + reader.u32(data + 4 * i) for i in range(n)]


def read_shot(reader: _Reader, table: int) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Return ``(edges, segments)`` where each segment is ``(wavelength, intensity)``."""
    edge_pos = reader.field(table, 0)
    seg_pos = reader.field(table, 1)
    if edge_pos is None or seg_pos is None:
        raise FlatBufferError("shot table missing edges or segments")
    edges = reader.doubles(edge_pos)

    n_seg, seg_data = reader.vector(seg_pos)
    if n_seg != N_SEGMENTS:
        raise FlatBufferError(f"expected {N_SEGMENTS} segments, got {n_seg}")

    segments = []
    for i in range(n_seg):
        p = seg_data + 4 * i
        seg_table = p + reader.u32(p)
        entries = reader.vtable(seg_table)
        if len(entries) < 2 or not all(entries[:2]):
            raise FlatBufferError(f"segment {i} missing calibration or data")
        cal = reader.doubles(seg_table + entries[0])
        intensity = reader.doubles(seg_table + entries[1])
        pixels = np.arange(intensity.size, dtype=float) + PIXEL_OFFSET
        wavelength = np.polyval(cal[::-1], pixels)
        segments.append((wavelength, intensity))
    return edges, segments


def stitch(edges: np.ndarray,
           segments: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the segments, each restricted to its slot in ``edges``.

    Returns ascending ``(wavelength, intensity)``.  The segments overlap on
    the detector, so a naive concatenation would double-cover ~30 nm at each
    seam; the edge list is the instrument's own choice of seam.
    """
    waves, amps = [], []
    for i, (wavelength, intensity) in enumerate(segments):
        lo, hi = float(edges[i]), float(edges[i + 1])
        keep = (wavelength >= lo) & (wavelength < hi)
        waves.append(wavelength[keep])
        amps.append(intensity[keep])
    wavelength = np.concatenate(waves)
    intensity = np.concatenate(amps)
    order = np.argsort(wavelength)
    return wavelength[order], intensity[order]


def to_export_grid(wavelength: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resample onto the vendor's 0.1 nm export grid (180.0 .. 960.9 nm).

    Outside the detector's coverage the instrument writes 0.0, so the same
    is done here rather than extrapolating.
    """
    grid = EXPORT_LO_NM + EXPORT_STEP_NM * np.arange(EXPORT_N)
    values = np.interp(grid, wavelength, intensity, left=0.0, right=0.0)
    return grid, values


def decode(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Decode every shot in a bundle to native-grid ``(wavelength, intensity)``."""
    reader = _Reader(open(path, "rb").read())
    out = []
    for table in _shot_tables(reader):
        edges, segments = read_shot(reader, table)
        out.append(stitch(edges, segments))
    return out


def write_csv(wavelength: np.ndarray, intensity: np.ndarray, path: str,
              export_grid: bool = True) -> None:
    if export_grid:
        wavelength, intensity = to_export_grid(wavelength, intensity)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wavelength", "intensity"])
        for x, y in zip(wavelength, intensity):
            writer.writerow([f"{x:.10g}", f"{y:.6g}"])


def read_vendor_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    wavelength, intensity = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue  # header
            wavelength.append(x)
            intensity.append(y)
    return np.asarray(wavelength), np.asarray(intensity)


def compare_with_csv(bundle: str, csv_path: str, shot: int) -> dict:
    """Compare one decoded shot against the vendor's exported CSV.

    This is the decoder's correctness check: the vendor export and this
    reconstruction come from the same detector counts, so on the shared
    grid they should agree to within resampling error.
    """
    shots = decode(bundle)
    if not 0 <= shot < len(shots):
        raise SystemExit(f"shot {shot} out of range (bundle has {len(shots)})")
    grid, mine = to_export_grid(*shots[shot])
    theirs_x, theirs = read_vendor_csv(csv_path)

    if theirs_x.size != grid.size or not np.allclose(theirs_x, grid, atol=1e-6):
        common = np.intersect1d(np.round(theirs_x, 4), np.round(grid, 4))
        mine = np.interp(common, grid, mine)
        theirs = np.interp(common, theirs_x, theirs)
        grid = common

    live = theirs != 0.0
    denom = float(np.max(np.abs(theirs))) or 1.0
    diff = mine - theirs
    with np.errstate(invalid="ignore"):
        corr = float(np.corrcoef(mine[live], theirs[live])[0, 1]) if live.sum() > 2 else float("nan")
    return {
        "n_points": int(grid.size),
        "n_nonzero_vendor": int(live.sum()),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "max_abs_diff_rel_pct": 100.0 * float(np.max(np.abs(diff))) / denom,
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
        "rms_diff_rel_pct": 100.0 * float(np.sqrt(np.mean(diff ** 2))) / denom,
        "correlation": corr,
        "vendor_max": float(np.max(theirs)),
        "decoded_max": float(np.max(mine)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", help="all_fb file from libzdata/spectra/")
    parser.add_argument("--out-dir", default=None, help="write per-shot CSVs here")
    parser.add_argument("--prefix", default="Shot", help="output filename prefix")
    parser.add_argument("--native-grid", action="store_true",
                        help="write the detector's own grid instead of the 0.1 nm export grid")
    parser.add_argument("--compare", default=None, help="vendor Shot(N).csv to validate against")
    parser.add_argument("--shot", type=int, default=0, help="shot index for --compare")
    args = parser.parse_args(argv)

    if args.compare:
        stats = compare_with_csv(args.bundle, args.compare, args.shot)
        width = max(len(k) for k in stats)
        for key, value in stats.items():
            print(f"  {key:<{width}s} : {value:.6g}" if isinstance(value, float)
                  else f"  {key:<{width}s} : {value}")
        return 0

    shots = decode(args.bundle)
    print(f"{len(shots)} shots in {os.path.basename(args.bundle)}")
    if not args.out_dir:
        for i, (wavelength, intensity) in enumerate(shots):
            print(f"  shot {i}: {wavelength.size} pts "
                  f"{wavelength.min():.2f}-{wavelength.max():.2f} nm "
                  f"max={intensity.max():.1f}")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    for i, (wavelength, intensity) in enumerate(shots):
        path = os.path.join(args.out_dir, f"{args.prefix}({i + 1}).csv")
        write_csv(wavelength, intensity, path, export_grid=not args.native_grid)
    print(f"wrote {len(shots)} CSVs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
