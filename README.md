# xyz-wfn-db
Collection of xyz structures and wfn files

The following set of structures were optimized at aug-cc-pvdz/pbe0 with
d3 dispersion, using pyscf coupled with the pyberny solver an and own
interface to grimme program available 

* molecules
* aromatic
* dimers
* tms

The diatomic set were optimized at the same functional level diatomic
but using a def2-tzvpd basis set.

All molecules in the test contain only singlet states, in some cases
like B2 or O2 these not correspond to the ground state. Carefull must be
taken with these potential problematic systems so in the case of dimers a
stability test was performed.

In a final step a T1 diagnostic was performed (althought this test can
not be fully realiable https://arxiv.org/abs/1806.05115) and the
potentially problematic or multirefence systems were separated inside
each folder in the multiref folder.

The isomer set and the pes-dimers contains files were taken from the MGCDB84
database (https://nmardirossian.wixsite.com/ks-dft/density-functionals)
except for the alanine and glycine isomers that were optimized at
MP2/cc-pvtz level 

