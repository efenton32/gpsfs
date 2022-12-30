from sim_demography import sim_param_change
from twosfs.spectra import Spectra, load_spectra
import matplotlib.pyplot as plt
import numpy as np

# ancestral_rates = [9.00e-4 for i in range(3)]
# new_rates = [7.00e-3 for i in range(3)]

# ancestral_rates = [1e-4, 1e-4, 1e-3]
# new_rates = [5e-3, 5e-3, 5e-3]

ancestral_rates = [1e-4, 1e-4, 1e-3]
new_rates = [5e-3, 5e-3, 5e-3]

param_dict = {"alpha": 1.20, "time":10, "samples": [36, 22, 5], "mig_rates": ancestral_rates,
              "new_rates": new_rates, "num_replicates":int(1e5)}

outfile = "test.hdf5"
save_path = "/n/home12/efenton/for_windows/gpsfs/"

sim_param_change(outfile, **param_dict)
spec = load_spectra(outfile)
spec_data = load_spectra("../2sfs/twosfs/agl_data/NC_044049/cod_NC_044049_initial_spectra.hdf5")

plt.figure()
plt.loglog(np.arange(126)[:63]+1, spec.normalized_onesfs(folded=True)[1:64], "x")
plt.loglog(np.arange(126)[:63]+1, spec_data.normalized_onesfs(folded=True)[1:64], "x")
plt.savefig(save_path + "param_change.pdf")
plt.close()


