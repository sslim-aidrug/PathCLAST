import os
import numpy as np

def mclust_R(x, n_clusters, model='EEE', random_seed=2020):
    os.environ['R_HOME'] = '/home/work/Root_STR1/Anaconda3/envs/path39/lib/R'
    os.environ['R_USER'] = '/home/work/Root_STR1/Anaconda3/envs/path39/lib/python3.9/site-packages/rpy2'

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(x), n_clusters, model)
    mclust_res = np.array(res[-2]).astype(int) - 1

    return mclust_res

def eval_mclust_ari(labels, z, n_clusters):
    raw_preds = mclust_R(z, n_clusters)
    preds = raw_preds[labels != -1]
    labels = labels[labels != -1]
    return raw_preds
