import numpy as np


def compute_inter_unq(a):
    key_list = list(a.keys())
    res = {}
    for key in key_list:
        res[key] = np.sum(np.array(a[key]))
    return res

a = np.load("fit_dict_Math_camel_gpt4o.pkl",allow_pickle=True)


inter_unq = compute_inter_unq(a)
np.save("inter_unq_Math_camel_gpt4o.npy",inter_unq)
