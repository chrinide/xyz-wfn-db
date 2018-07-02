#!/usr/bin/env python

import numpy, sys, os
from pyscf import lib, gto, scf, dft
from pyscf.tools import molden
from pyscf.data import radii

def read_molecule(path):

    charge = spin = 0
    with open(path, 'r') as myfile:
        output = myfile.read()
        output = output.lstrip()
        output = output.rstrip()
        output = output.split('\n')

    try:
        int(output[0])
    except ValueError:
        try:
            charge = int(output[0].split(' ')[0])
            spin = int(output[0].split(' ')[1]) - 1
        except ValueError:
            molecule = output
        else:
            molecule = '\n'.join(output[1:])
    else:
        if int(output[0]) == len(output) - 2:
            molecule = '\n'.join(output[2:])
            try:
                charge = int(output[1].split(' ')[0])
                spin = int(output[1].split(' ')[1])-1
            except ValueError:
                pass
        else:
            print "THIS IS NOT A VALID XYZ FILE"

    return (molecule, charge, spin)

directory = './'
files = [i for i in os.listdir(directory) if i.endswith('.xyz')]

for filename in files:

    name = filename[:-4]
    mol = gto.Mole()
    (mol.atom, charge, spin) = read_molecule(directory+filename)
    mol.basis = 'aug-cc-pvtz'
    mol.charge = charge
    mol.spin = spin
    mol.output = name+'.out'
    mol.symmetry = 0
    mol.max_memory = 10000
    mol.verbose = 4
    mol.build()

    xyzfile = name + '.xyz'
    fspt = open(xyzfile,'w')
    coords = mol.atom_coords()*lib.param.BOHR
    fspt.write('%d \n' % mol.natm)
    fspt.write('%d %d\n' % (mol.charge, (mol.spin+1)))
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        fspt.write('%s  %12.6f  %12.6f  %12.6f\n' % (symb, \
        coords[ia][0],coords[ia][1], coords[ia][2])) 
