import numpy as np
from scipy.stats import norm

def ks_distance(v1, v2):
    cdf1 = np.cumsum(v1/np.sum(v1, axis=0, keepdims=True), axis=0)
    cdf2 = np.cumsum(v2/np.sum(v2, axis=0, keepdims=True), axis=0)
    return np.max(np.abs(cdf1 - cdf2), axis=0)

def summed_ratio(v1, v2):
    return sum(np.max([v1/v2, v2/v1], axis = 0))

def compare_spectra(spec1, spec2, method = "ks"):
    if method == "ks":
        weighting = np.arange(len(spec1.normalized_onesfs(folded=True))) + 1
        v1 = spec1.normalized_onesfs(folded=True) * weighting
        v2 = spec2.normalized_onesfs(folded=True) * weighting
        return ks_distance(v1, v2)
    elif method == "summed_ratio":
        v1 = spec1.normalized_onesfs(folded=True)[1:64]
        v2 = spec2.normalized_onesfs(folded=True)[1:64]
        return summed_ratio(v1, v2)

def find_global_max(f, model, bounds, dx, n_restarts=10, n_steps = 1000, decimation = 1000, **params):
    bounds_range = np.diff(bounds, axis = 1)
    n_params = bounds.shape[0]
    best_pt = np.zeros(n_params)
    best_val = -10
    for restart in range(n_restarts):
        curr_pt = np.random.random((n_params,1))*bounds_range + bounds[:,0].reshape((-1, 1))
        curr_val = f(model, curr_pt.T, **params)
        for step in range(n_steps):
            slope = np.zeros((n_params, 1))
            for p in range(n_params):
                test_pt = curr_pt + bounds_range[p,0] / decimation
                test_val = f(model, test_pt.T, **params)
                slope[p,0] = (test_val - curr_val) / (bounds_range[p,0] / decimation)
            curr_pt = curr_pt + slope * dx
            curr_pt = np.max([curr_pt, bounds[:,0,None]], axis = 0)
            curr_pt = np.min([curr_pt, bounds[:,1,None]], axis = 0)
            curr_val = f(model, curr_pt.T, **params)
        if curr_val > best_val:
            best_pt = curr_pt
            best_val = curr_val
    return best_pt[:,0]

def prob_of_improvement(model, params, best):
    ks_pred, std = model.predict(params, return_std = True)
    p_improve = norm.cdf(best, ks_pred, std)
    return p_improve[0]

def expected_val(model, params):
    ks_pred = model.predict(params, return_std = False)
    return ks_pred

def undert(model, params):
    ks_pred, std = model.predict(params, return_std = True)
    return std

def linearize(val, the_max, the_min):
    return (np.log(val)-np.log(the_min)) / (np.log(the_max) - np.log(the_min))

def logify(val, the_max, the_min):
    return np.exp(val * (np.log(the_max) - np.log(the_min)) + np.log(the_min))

def dict_to_list(param_dict):
    params = []
    for x in param_dict.values():
        if type(x) == list:
            for y in x:
                params.append(round(y, 10))
        else:
            params.append(round(x, 10))
    return params

def list_to_dict(param_list, keys, lengths):
    x = 0
    num_keys = len(keys)
    param_dict = {}
    for k, l in zip(keys, lengths):
        if l == 1:
            param_dict[k] = round(param_list[x], 6)
        else:
            param_dict[k] = [round(p, 6) for p in param_list[x:x+l]]
        x += l
    return param_dict

def build_samps(samp_size):
    x = [[a,b,samp_size-a-b] 
         for a in range(int(np.ceil(samp_size/3)), samp_size+1)
         for b in range(int(np.ceil((samp_size-a)/2)), min(a+1, samp_size - a))]
    return x


