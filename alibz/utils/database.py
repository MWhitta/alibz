import pickle
import csv
import os
import sysconfig
from pathlib import Path

import numpy as np

from alibz.utils.wavelength import vacuum_to_air

class Database():
    """ class containing the lines and ionization states of an element imported from database
    """

    # The output/index ordering remains the full H--U sequence.  These five
    # positions are deliberately unavailable for the current development
    # round; callers must not reinterpret them as measured zeros.
    UNSUPPORTED_ELEMENTS = frozenset({'Pm', 'Po', 'At', 'Rn', 'Pa'})

    @staticmethod
    def _resolve_dbpath(dbpath="db"):
        requested = "db" if dbpath is None else str(dbpath)
        default_db = requested in {"db", "./db"}

        if not default_db:
            candidate = Path(requested).expanduser()
            if candidate.is_dir():
                return candidate.absolute()
            raise FileNotFoundError(
                f"Database path {dbpath!r} was not found"
            )

        env_db = os.environ.get("ALIBZ_DB")
        if env_db:
            candidate = Path(env_db).expanduser()
            if candidate.is_dir():
                return candidate.absolute()
            raise FileNotFoundError(
                f"ALIBZ_DB points to a missing database directory: {env_db!r}"
            )

        project_root = Path(__file__).resolve().parents[2]
        data_root = Path(sysconfig.get_path("data") or sysconfig.get_path("prefix"))
        candidates = [
            Path(requested).expanduser(),
            project_root / "db",
            project_root / "share" / "alibz" / "db",
            data_root / "share" / "alibz" / "db",
        ]

        for candidate in candidates:
            if candidate.is_dir():
                return candidate.absolute()

        raise FileNotFoundError(
            "Database path 'db' was not found. Searched ./db, the source "
            "checkout db, and the installed share/alibz/db. Pass an explicit "
            "dbpath or set ALIBZ_DB."
        )

    def __init__(self, dbpath) -> None:
        dbpath = self._resolve_dbpath(dbpath)
        self.dbpath = dbpath
        self.elements= ['H', 'He', #row1
                        'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', #row2
                        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', #row3
                        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', #row4
                        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', #row5
                        'Cs', 'Ba', #row6 alkali/alkaline earth
                        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', #row6 rare earths
                        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', #row6 transition metals
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U'] #row 7 stable actinide elements

        #database missing data for these elements
        with open(self.dbpath / "no_lines26.pickle", 'rb') as f:
            self.no_lines = pickle.load(f)

        # Physical isotope stability and database support are distinct.  The
        # former is useful metadata; the latter is the gate analysis code must
        # use.  Se, Th and U are supported, while five very-low-abundance
        # radionuclides retain explicit output positions but do not
        # participate in fitting or synthetic sampling this round.
        self.unstable_elements = {
            'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa',
        }
        # Default real-sample analysis assumes naturally occurring targets.
        # This is intentionally stricter than synthetic/schema support.
        self.analysis_excluded_elements = set(self.unstable_elements)
        self.unsupported_elements = set(self.UNSUPPORTED_ELEMENTS)
        # These elements are available to the forward/inverse model, but most
        # public oscillator strengths are older intensity-derived estimates.
        # They must not define wavelength-calibration anchors.
        self.strength_uncertain_elements = {'Se', 'Th', 'U'}
        self.supported_elements = tuple(
            el for el in self.elements if el not in self.unsupported_elements
        )
        self.support_mask = np.array(
            [el not in self.unsupported_elements for el in self.elements],
            dtype=bool,
        )
        
        with open(self.dbpath / "el_lines92.pickle", 'rb') as f:
            self.atom_dict = pickle.load(f)

        # NIST observed/classified lines that lack enough fields for the
        # quantitative forward model live in a separate, lazy-loaded catalog.
        # They are evidence for identification/QC, never implicit gA values.
        self.observed_line_path = self.dbpath / "observed_lines_nist.tsv"
        self._observed_line_dict = None

        # The pickled line lists hold Ritz VACUUM wavelengths, but observed
        # spectra are air-calibrated (ASD convention: air above 200 nm).
        # Convert once at load so every consumer — forward synthesis and
        # inverse indexing alike — works in air wavelengths.  Unconverted,
        # the 0.11-0.24 nm vacuum-air offset exceeds the indexer's matching
        # tolerance and observed peaks silently match wrong lines.
        for el, arr in self.atom_dict.items():
            if arr.size == 0:
                continue
            air = vacuum_to_air(arr[:, 1].astype(float))
            arr[:, 1] = np.char.mod('%.6f', air)

        # ionization energies
        with open(self.dbpath / "ionization" / "ionization.pickle", 'rb') as f:
            self.ion= pickle.load(f)

        #relative natural abundance of elements
        abund = np.loadtxt(self.dbpath / "abundance_92.csv") # crustal elemental abundance
        self.elem_abund = abund / np.sum(abund) # normalized elemental abundance probability
        self.elem_abund = {i: j for i, j in zip(self.elements, self.elem_abund)} # make dictionary from list

    def lines(self, el, ion=0):
        lines_array = self.atom_dict[el]
        if ion:
            ion_lines = lines_array[:,0].astype(float).astype(int) == ion
            lines_array = lines_array[ion_lines]
        return lines_array

    def ionization_energy(self, el, ion=0):
        ionization_array = self.ion[el]
        if ion:
            ion_stage = int(ion)
            ion_mask = ionization_array[:, 1].astype(float).astype(int) == ion_stage - 1
            ionization_array = ionization_array[ion_mask]
        return ionization_array
    
    def abundance(self, el):
        abundance_val = self.elem_abund[el]
        return abundance_val

    def is_supported(self, el):
        """Whether an H--U output position participates in this model round."""
        if el not in self.elements:
            raise KeyError(el)
        return el not in self.unsupported_elements

    def observed_lines(self, el, ion=0):
        """Return NIST observed-line evidence, including non-quantitative rows.

        Wavelengths are served in the same vacuum-below-200/air-above-200
        convention as :meth:`lines`.  Each record carries a
        ``quantitative_ready`` boolean; callers must not infer missing
        transition probabilities from observed intensity.
        """
        if el not in self.elements:
            raise KeyError(el)
        if self._observed_line_dict is None:
            self._observed_line_dict = {element: [] for element in self.elements}
            if self.observed_line_path.exists():
                with self.observed_line_path.open(newline="") as handle:
                    for raw in csv.DictReader(handle, delimiter="\t"):
                        element = raw["element"]
                        vacuum_nm = float(raw["wavelength_vacuum_nm"])
                        record = dict(raw)
                        record["ion_stage"] = int(raw["ion_stage"])
                        record["wavelength_nm"] = float(vacuum_to_air(vacuum_nm))
                        record["quantitative_ready"] = raw["quantitative_ready"] == "1"
                        for key in ("gA_s-1", "Ei_eV", "Ek_eV", "g_i", "g_k"):
                            record[key] = float(raw[key]) if raw[key] else None
                        self._observed_line_dict.setdefault(element, []).append(record)
        records = self._observed_line_dict.get(el, [])
        if ion:
            return [row for row in records if row["ion_stage"] == int(ion)]
        return list(records)
