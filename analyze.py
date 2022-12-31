import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, load_spectra
import json
import sys

save_path = "/n/home12/efenton/for_windows/gpsfs/"
load_path = "sims/NC_044049/"
sim_file = "spec_batch_{}_rep_{}.hdf5"
data_file = "../2sfs/twosfs/agl_data/NC_044049/cod_NC_044049_initial_spectra.hdf5"
num_sims = 20

if sys.argv[1] == "idx":
    idx = int(sys.argv[2])
elif sys.argv[1] == "batch":
    ks = []
    with open(load_path + "results_{}.jsonl".format(sys.argv[2])) as df:
        for line in df:
            try:
                ks.append(float(json.loads(line)["ks"]))
            except:
                pass
    idx = np.argmin(ks)
else:
    raise ValueError("Must be either an idx or batch")

batch = idx // num_sims + 1
rep = idx % num_sims

print("batch " + str(batch) + ", rep " + str(rep))

spec_sim = load_spectra(load_path + sim_file.format(batch, rep))
spec_data = load_spectra(data_file)

plt.figure()
plt.loglog(np.arange(126)[:63] + 1, spec_sim.normalized_onesfs(folded=True)[1:64], "x", label = "sim")
plt.loglog(np.arange(126)[:63] + 1, spec_data.normalized_onesfs(folded=True)[1:64], "x", label = "data")
plt.legend()
plt.savefig(save_path + "test_param_change.pdf")
plt.close()

