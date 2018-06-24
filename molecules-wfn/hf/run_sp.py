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

directory = '../../diatomic/'
files = [i for i in os.listdir(directory) if i.endswith('.xyz')]

for filename in files:

    name = filename[:-8]
    mol = gto.Mole()
    (mol.atom, charge, spin) = read_molecule(directory+filename)
    mol.basis = 'def2-qzvpd'
    mol.charge = charge
    mol.spin = spin
    mol.output = name+'.out'
    mol.symmetry = 0
    mol.verbose = 4
    mol.build()

    mf = scf.RHF(mol) 
    mf = scf.addons.remove_linear_dep_(mf)
    mf.max_cycle = 120
    mf = mf.newton()
    mf.chkfile = name+'.chk'
    mf.kernel()

    stable_cyc = 3
    for i in range(stable_cyc):
        new_mo_coeff = mf.stability(internal=True, external=False)[0]
        if numpy.linalg.norm(numpy.array(new_mo_coeff) - numpy.array(mf.mo_coeff)) < 10**-8:
            lib.logger.info(mf,"* The molecule is stable")
            break
        else:
            lib.logger.info(mf,"* The molecule is unstable")
            n_alpha = numpy.count_nonzero(mf.mo_occ)
            P_alpha = 2.0*numpy.dot(new_mo_coeff[:, :n_alpha], new_mo_coeff.T[:n_alpha])
            mf.kernel(dm0=(P_alpha))
            lib.logger.info(mf,"* Updated SCF energy and orbitals: %16.f" % mf.e_tot)
     
    coeff = mf.mo_coeff[:,mf.mo_occ>0]
    occ = mf.mo_occ[mf.mo_occ>0]
    energy = mf.mo_energy[mf.mo_occ>0]
    den_file = name+'.mol'
    fspt = open(den_file,'w')
    molden.header(mol, fspt)
    molden.orbital_coeff(mol, fspt, coeff, ene=energy, occ=occ) 
    fspt.close()                    
    cmd = '/home/jluis/bin/molden2aim '+name
    os.system(cmd)
    den_file = name+'.wfn'
    fspt = open(den_file,'a')
    fspt.write('RHF\n')
    fspt.close()                    

    pmd_file = name+'.pmd'
    fspt = open(pmd_file,'w')
    fspt.write('%s.wfn\n' % (name))
    if (mol.symmetry == False):
        fspt.write('nosymmetry\n')
    fspt.write('tes\n')
    fspt.write('  epsiscp 0.220\n')
    fspt.write('  radialquad 7\n')
    fspt.write('  rmapping 2\n')
    fspt.write('  lmax 10\n')
    fspt.write('  nr 551\n')
    fspt.write('  lebedev 5810\n')
    fspt.write('  betasphere\n')
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        inuc = lib.parameters.NUC[symb]
        betarad = radii.COVALENT[inuc]/2.5
        fspt.write('  betaradw %d %1.3f\n' % ((ia+1),betarad))
    fspt.write('  radialquadbeta 7\n')
    fspt.write('  rmappingbeta 3\n')
    fspt.write('  lmaxbeta 8\n')
    fspt.write('  nrb 451\n')
    fspt.write('  lebedevbeta 3074\n')
    fspt.write('  dafh\n')
    fspt.write('endtes')
    fspt.close()                    

    cmd = 'rm '+name+'.mol'
    os.system(cmd)
