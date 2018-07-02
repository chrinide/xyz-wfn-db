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

directory = '../../aromatic/'
files = [i for i in os.listdir(directory) if i.endswith('.xyz')]

for filename in files:

    name = filename[:-8]
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

    mf = dft.RKS(mol) 
    mf.grids.level = 4
    mf.xc = 'pbe'
    mf.max_cycle = 120
    mf.chkfile = name+'.chk'
    mf = scf.addons.remove_linear_dep_(mf)
    mf.kernel()

    dm = mf.make_rdm1()
    nao = mol.nao_nr()
    unit = 2.541746
    origin = ([0.0,0.0,0.0])
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    mol.set_common_orig(origin)
    r2 = mol.intor_symmetric('int1e_r2')
    r2 = numpy.einsum('ij,ji->', r2, dm)
    lib.logger.info(mf,'Electronic spatial extent <R**2> (au): %.4f', r2)

    lib.logger.info(mf,'* Multipoles in the independent field-basis, Gauge -> (0,0,0)')
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = numpy.einsum('xij,ji->x', ao_dip, dm)
    lib.logger.info(mf,'Electronic Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *el_dip*unit)
    nucl_dip = numpy.einsum('i,ix->x', charges, coords)
    lib.logger.info(mf,'Nuclear Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *nucl_dip*unit)
    mol_dip = (nucl_dip - el_dip) * unit
    lib.logger.info(mf,'Total Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *mol_dip)

    lib.logger.info(mf,'Quadrupole moments (Debye-Angs)')
    rr = mol.intor_symmetric('int1e_rr', comp=9).reshape(3,3,nao,nao)
    rr = -1.0*numpy.einsum('xyij,ji->xy', rr, dm)
    rr += numpy.einsum('z,zx,zy->xy', charges, coords, coords)
    rr = rr*unit*lib.param.BOHR
    lib.logger.info(mf,'Total Quadrupole moments (XX, YY, ZZ): %.4f, %.4f, %.4f', \
    rr[0,0], rr[1,1], rr[2,2])
    lib.logger.info(mf,'Total Quadrupole moments (XY, XZ, YZ): %.4f, %.4f, %.4f', \
    rr[0,1], rr[0,2], rr[1,2])
 
    lib.logger.info(mf,'Octupole moments (Debye-Angs**2)')
    rrr = mol.intor_symmetric('int1e_rrr', comp=27).reshape(3,3,3,nao,nao)
    rrr = -1.0*numpy.einsum('xyzij,ji->xyz', rrr, dm)
    rrr += numpy.einsum('z,zx,zy,zk->xyk', charges, coords, coords, coords)
    rrr = rrr*unit*lib.param.BOHR**2
    lib.logger.info(mf,'Total Octupole moments (XXX, YYY, ZZZ, XYY): %.4f, %.4f, %.4f, %.4f', \
    rrr[0,0,0], rrr[1,1,1], rrr[2,2,2], rrr[0,1,1])
    lib.logger.info(mf,'Total Octupole moments (XXY, XXZ, XZZ, YZZ): %.4f, %.4f, %.4f, %.4f', \
    rrr[0,0,1], rrr[0,0,2], rrr[0,2,2], rrr[1,2,2])
    lib.logger.info(mf,'Total Octupole moments (YYZ, XYZ): %.4f, %.4f', rrr[1,1,2], rrr[0,1,2])

    lib.logger.info(mf,'Hexadecapole moments (Debye-Angs**3)')
    rrrr = mol.intor_symmetric('int1e_rrrr', comp=81).reshape(3,3,3,3,nao,nao)
    rrrr = -1.0*numpy.einsum('xyzwij,ji->xyzw', rrrr, dm)
    rrrr += numpy.einsum('z,zx,zy,zk,zw->xykw', charges, coords, coords, coords, coords)
    rrrr = rrrr*unit*lib.param.BOHR**3
    lib.logger.info(mf,'Total Hexadecapole moments (XXXX, YYYY, ZZZZ, XXXY): %.4f, %.4f, %.4f, %.4f', \
    rrrr[0,0,0,0], rrrr[1,1,1,1], rrrr[2,2,2,2], rrrr[0,0,0,1])
    lib.logger.info(mf,'Total Hexadecapole moments (XXXZ, YYYX, YYYZ, ZZZX): %.4f, %.4f, %.4f, %.4f', \
    rrrr[0,0,0,2], rrrr[1,1,1,0], rrrr[1,1,1,2], rrrr[2,2,2,0])
    lib.logger.info(mf,'Total Hexadecapole moments (ZZZY, XXYY, XXZZ, YYZZ): %.4f, %.4f, %.4f, %.4f', \
    rrrr[2,2,2,1], rrrr[0,0,1,1], rrrr[0,0,2,2], rrrr[1,1,2,2])
    lib.logger.info(mf,'Total Hexadecapole moments (XXYZ, YYXZ, ZZXY): %.4f, %.4f, %.4f', \
    rrrr[0,0,1,2], rrrr[1,1,0,2], rrrr[2,2,0,1])
     
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
        betarad = radii.COVALENT[inuc]/3.5
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

