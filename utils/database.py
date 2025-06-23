import pickle
import numpy as np

class Database():
    """ class containing the lines and ionization states of an element imported from database
    """

    def __init__(self, dbpath) -> None:
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
        with open(f"{dbpath}/no_lines26.pickle", 'rb') as f:
            self.no_lines = pickle.load(f)
        
        with open(f"{dbpath}/el_lines92.pickle", 'rb') as f:
            self.atom_dict = pickle.load(f)

        # ionization energies
        with open(f"{dbpath}/ionization/ionization.pickle", 'rb') as f:
            self.ion= pickle.load(f)

        #relative natural abundance of elements
        abund = np.loadtxt(f"{dbpath}/abundance_92.csv") # crustal elemental abundance
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
            ion_lines = lines_array[:,0].astype(float).astype(int) == ion
            lines_array = lines_array[ion_lines]
        return ionization_array
    
    def abundance(self, el):
        abundance_val = self.elem_abund[el]
        return abundance_val