import numpy as np
import matplotlib.pyplot as plt
import msprime
from twosfs.spectra import Spectra, add_spectra, spectra_from_TreeSequence
from itertools import combinations

def sim_demography(save_file, alpha=2.0, samples=[64,63], mig_rates=[0.01], num_replicates=int(1e4)):
    demes = len(samples)
    demography = msprime.Demography()
    for i in range(demes):
        demography.add_population(name="pop_"+str(i), initial_size=100)

    pairs = combinations(["pop_"+str(i) for i in range(demes)], 2)
    for pair, rate in zip(pairs, mig_rates):
        demography.set_symmetric_migration_rate(pair, rate)

    samples_dict = {"pop_"+str(i): samples[i] for i in range(demes)}

    if alpha == 2.0:
        model = None
    else:
        model = msprime.BetaCoalescent(alpha = alpha)

    sims = msprime.sim_ancestry(
        samples = samples_dict,
        demography = demography,
        sequence_length = 1,
        model = model,
        recombination_rate = 0,
        num_replicates = num_replicates,
    )

    windows = [0, 1]
    spec = add_spectra(spectra_from_TreeSequence(windows, 0, tseq) for tseq in sims)
    spec.save(save_file)



def sim_admixture(save_file, alpha=2.0, time=0, ratio=[64,63], mig_rates=[0.01], num_replicates=int(1e4)):
    demes = len(ratio)
    demography = msprime.Demography()
    for i in range(demes):
        demography.add_population(name="pop_"+str(i), initial_size=100)
    demography.add_population(name="Sampled", initial_size=1)

    pairs = combinations(["pop_"+str(i) for i in range(demes)], 2)
    for pair, rate in zip(pairs, mig_rates):
        demography.set_symmetric_migration_rate(pair, rate)

    num_samps = sum(ratio)
    ratio_normed = [r / num_samps for r in ratio]
    ancestral = ["pop_"+str(i) for i in range(demes)]

    demography.add_admixture(time = time, derived = "Sampled", ancestral = ancestral, proportions = ratio_normed)

    if alpha == 2.0:
        model = None
    else:
        model = msprime.BetaCoalescent(alpha = alpha)

    samples_dict = {"Sampled": num_samps}

    sims = msprime.sim_ancestry(
        samples = samples_dict,
        demography = demography,
        sequence_length = 1,
        model = model,
        recombination_rate = 0,
        num_replicates = num_replicates,
    )

    windows = [0, 1]
    spec = add_spectra(spectra_from_TreeSequence(windows, 0, tseq) for tseq in sims)
    spec.save(save_file)


def sim_param_change(save_file, alpha=2.0, time=1.0, samples=[32, 31], anc_rates=[0.01],
                     new_rates=[0.01], num_replicates=int(1e4)):

    # Define the demes and demography
    demes = len(samples)
    demography = msprime.Demography()
    for i in range(demes):
        demography.add_population(name="pop_"+str(i), initial_size=100)

    # Define migration parameters
    pairs = combinations(["pop_"+str(i) for i in range(demes)], 2)
    for pair, rate1, rate2 in zip(pairs, anc_rates, new_rates):
        demography.set_symmetric_migration_rate(pair, rate2)
        demography.add_migration_rate_change(time=time, rate=rate1, source=pair[0], dest=pair[1])
        demography.add_migration_rate_change(time=time, rate=rate1, source=pair[1], dest=pair[0])

    # Define the coalescent model
    if alpha == 2.0:
        model = None
    else:
        model = msprime.BetaCoalescent(alpha = alpha)

    # How many samples come from each deme and sim
    samples_dict = {"pop_"+str(i): samples[i] for i in range(demes)}
    sims = msprime.sim_ancestry(
        samples = samples_dict,
        demography = demography,
        sequence_length = 1,
        model = model,
        recombination_rate = 0,
        num_replicates = num_replicates,
    )

    # Exctract spectra and save
    windows = [0, 1]
    spec = add_spectra(spectra_from_TreeSequence(windows, 0, tseq) for tseq in sims)
    spec.save(save_file)

