#!/usr/bin/env python

import os, time, numpy, ctypes
from pyscf import gto, scf, lib, dft

_loaderpath = os.path.dirname('/home/jluis/src/git/scripts/dft/post-vv10/')
libvv10 = numpy.ctypeslib.load_library('libvv10.so', _loaderpath)

chkfile = 'test.chk'

mol = lib.chkfile.load_mol(chkfile)
mo_coeff = scf.chkfile.load(chkfile, 'scf/mo_coeff')
mo_occ = scf.chkfile.load(chkfile, 'scf/mo_occ')
dm = numpy.dot(mo_coeff*mo_occ, mo_coeff.T)

grids = dft.Grids(mol)
grids.verbose = 4
grids.level = 4
grids.build()
coords = grids.coords
weights = grids.weights
ngrids = len(weights)
lib.logger.info(mol, 'Size of XC grid %d', ngrids)

nlgrids = dft.Grids(mol)
nlgrids.verbose = 4
nlgrids.level = 2
nlgrids.prune = dft.gen_grid.sg1_prune
nlgrids.build()
nlcoords = nlgrids.coords
nlweights = nlgrids.weights
nlngrids = len(nlweights)
lib.logger.info(mol, 'Size of NL grid %d', nlngrids)

#nao, nmo = mo_coeff.shape
#s = mol.intor('int1e_ovlp')
#t = mol.intor('int1e_kin')
#v = mol.intor('int1e_nuc')
#eri_ao = ao2mo.restore(1,mol.intor('int2e'))
#eri_ao = eri_ao.reshape(nao,nao,nao,nao)

#enuc = mol.energy_nuc() 
#ekin = numpy.einsum('ij,ji->',t,dm)
#pop = numpy.einsum('ij,ji->',s,dm)
#elnuce = numpy.einsum('ij,ji->',v,dm)
#lib.logger.info(mol, 'Population : %12.6f' % pop)
#lib.logger.info(mol, 'Kinetic energy : %12.6f' % ekin)
#lib.logger.info(mol, 'Nuclear Atraction energy : %12.6f' % elnuce)
#lib.logger.info(mol, 'Nuclear Repulsion energy : %12.6f' % enuc)
#bie1 = numpy.einsum('ijkl,ij,kl->',eri_ao,dm,dm)*0.5 # J
#bie2 = numpy.einsum('ijkl,il,jk->',eri_ao,dm,dm)*0.25 # XC
#pairs1 = numpy.einsum('ij,kl,ij,kl->',dm,dm,s,s) # J
#pairs2 = numpy.einsum('ij,kl,li,kj->',dm,dm,s,s)*0.5 # XC
#pairs = (pairs1 - pairs2)
#lib.logger.info(mol, 'Coulomb Pairs : %12.6f' % (pairs1))
#lib.logger.info(mol, 'XC Pairs : %12.6f' % (pairs2))
#lib.logger.info(mol, 'Pairs : %12.6f' % pairs)
#lib.logger.info(mol, 'J energy : %12.6f' % bie1)
#lib.logger.info(mol, 'XC energy : %12.6f' % -bie2)
#lib.logger.info(mol, 'EE energy : %12.6f' % (bie1-bie2))
#etot = enuc + ekin + elnuce + bie1 - bie2
#lib.logger.info(mol, 'Total energy : %12.6f' % etot)

ao = dft.numint.eval_ao(mol, coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
gnorm2 = numpy.zeros(ngrids)
for i in range(ngrids):
    gnorm2[i] = numpy.linalg.norm(rho[-3:,i])**2
lib.logger.info(mol ,'Rho = %.12f' % numpy.einsum('i,i->', rho[0], weights))
ex, vx = dft.libxc.eval_xc('rPW86,', rho)[:2]
ec, vc = dft.libxc.eval_xc(',PBE', rho)[:2]
lib.logger.info(mol, 'Exc = %.12f' % numpy.einsum('i,i,i->', ex+ec, rho[0], weights))

#t = time.time()
#libvv10.vv10(ctypes.c_int(ngrids),
#             coords.ctypes.data_as(ctypes.c_void_p),
#             rho.ctypes.data_as(ctypes.c_void_p),
#             weights.ctypes.data_as(ctypes.c_void_p),
#             gnorm2.ctypes.data_as(ctypes.c_void_p))
#lib.logger.info(mol, 'Total time taken VV10 : %.3f seconds' % (time.time()-t))
