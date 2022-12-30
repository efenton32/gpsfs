from sim_demography import sim_demography, sim_admixture
from twosfs.spectra import Spectra, load_spectra
import matplotlib.pyplot as plt
import numpy as np

rates = [2e-4]
param_dict = {"alpha": 1.05, "time":1.0, "ratio": [36, 27], "mig_rates": rates, "num_replicates":int(1e5)}

outfile = "test.hdf5"
save_path = "/n/home12/efenton/for_windows/gpsfs/"

sim_admixture(outfile, **param_dict)
spec = load_spectra(outfile)
spec_data = load_spectra("../2sfs/twosfs/agl_data/NC_044049/cod_NC_044049_initial_spectra.hdf5")

plt.figure()
plt.loglog(np.arange(126)[:63]+1, spec.normalized_onesfs(folded=True)[1:64], "x")
plt.loglog(np.arange(126)[:63]+1, spec_data.normalized_onesfs(folded=True)[1:64], "x")
plt.savefig(save_path + "admixed.pdf")
plt.close()


