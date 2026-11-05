#!/usr/bin/env python3
"""Build db/partition_barklem2016.npz from the VizieR ASU TSV of
J/A+A/588/A96/table8 (Barklem & Collet 2016, atomic partition functions).

Fetch (the CDS ftp mirrors timed out; the VizieR-CfA ASU service works):
  curl -L 'http://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/A%2bA/588/A96/table8&-out.max=unlimited&-out.all=2&-out.meta=hud' -o table8.tsv
The '#Column ... T=<value>' header lines give the 42-point T grid (K); each
data row is 'El_ROMAN  Q(T1) ... Q(T42)'.  Output arrays: species (str),
T_K (42,), lnU (n_species, 42).
"""
import argparse, hashlib, re, sys
import numpy as np


def parse(path):
    T, species, rows = [], [], []
    for line in open(path):
        if line.startswith("#Column"):
            m = re.search(r"T=([0-9.eE+-]+)", line)
            if m:
                T.append(float(m.group(1)))
            else:  # 'T=' is cut off by the tab-split for pf columns > 1; fall back
                pass
            continue
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 43 or not re.match(r"^\s*[A-Z][a-z]?_[IVX]+\s*$", parts[0]):
            continue
        species.append(parts[0].strip())
        rows.append([float(x) for x in parts[1:43]])
    return np.array(T), species, np.array(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv"); ap.add_argument("out")
    a = ap.parse_args(argv)
    txt = open(a.tsv).read()
    T = np.array([float(x) for x in re.findall(r"^#Column\tpf\d+\t.*?T=([0-9.eE+-]+)", txt, re.M)])
    if T.size == 0:  # meta lines differ: header rows of the form '\tT=...'
        T = np.array([float(x) for x in re.findall(r"T=([0-9.eE+-]+)", txt)])
    _, species, Q = parse(a.tsv)
    assert T.size == 42 and Q.shape[1] == 42, (T.size, Q.shape)
    assert np.all(Q > 0) and np.all(np.diff(T) > 0)
    np.savez_compressed(a.out, species=np.array(species), T_K=T, lnU=np.log(Q))
    print(a.out, len(species), "species; T", T[0], "-", T[-1],
          "sha256(tsv)=" + hashlib.sha256(txt.encode()).hexdigest())


if __name__ == "__main__":
    main()
