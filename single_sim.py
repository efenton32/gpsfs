from sim_demography import sim_demography
from twosfs.spectra import Spectra, load_spectra
import matplotlib.pyplot as plt
import numpy as np

def fuzz(sfs, level = 0.0):
    fuzzer = sfs * level
    sfs[:-1] += fuzzer[1:]
    sfs[1:] += fuzzer[:-1]
    return sfs

a = 1e-3
rates = [1e-3, a, a]
param_dict = {"alpha": 1.1, "samples": [40, 23, 0], "mig_rates": rates, "num_replicates":int(1e5)}

# rates = [1e-3]
# param_dict = {"alpha": 2.0, "samples": [32, 31], "mig_rates": rates, "num_replicates":int(1e4)}

outfile = "test.hdf5"
save_path = "/n/home12/efenton/for_windows/gpsfs/"

sim_demography(outfile, **param_dict)
spec = load_spectra(outfile)

plt.figure()
plt.loglog(np.arange(126)+1, spec.normalized_onesfs(folded=True)[1:], "x")
# plt.loglog((1, 64), (.1, .1/64), color="k")
plt.savefig(save_path + "single.pdf")
plt.close()


