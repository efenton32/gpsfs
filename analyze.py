import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, load_spectra
import json

save_path = "/n/home12/efenton/for_windows/gpsfs/"
load_path = "sims/NC_044049/"
sim_file = "spec_batch_{}_rep_{}.hdf5"
data_file = "../2sfs/twosfs/agl_data/NC_044049/cod_NC_044049_initial_spectra.hdf5"

ks = []
with open(load_path + "results_7.jsonl") as df:
    for line in df:
        try:
            ks.append(float(json.loads(line)["ks"]))
        except:
            pass

idx = np.argmin(ks)

batch = idx // 100 + 1
rep = idx % 100

print(idx)
print(batch)
print(rep)

spec_sim = load_spectra(load_path + sim_file.format(batch, rep))
spec_data = load_spectra(data_file)

plt.figure()
plt.loglog(np.arange(126) + 1, spec_data.normalized_onesfs(folded=True)[1:], "x", label = "data")
plt.loglog(np.arange(126) + 1, spec_sim.normalized_onesfs(folded=True)[1:], "x", label = "sim")
plt.legend()
plt.savefig(save_path + "test1.png")
plt.close()

