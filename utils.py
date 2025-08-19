import os
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import cv2
from skimage.feature import graycomatrix, graycoprops
from tqdm import trange
import numba
from scipy.sparse import issparse
from metrics import eval_mclust_ari
from sklearn.metrics import adjusted_rand_score

def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if load_images is False:
        if file_adj is None:
            file_adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)
        sc.pp.normalize_total(adata_h5, target_sum=1e4)
        sc.pp.log1p(adata_h5)
        sc.pp.scale(adata_h5, zero_center=True, max_value=None)
    return adata_h5


def build_her2st_data(path, name, size=112):
    cnt_path = os.path.join(path, 'data/ST-cnts', f'{name}.tsv')
    df_cnt = pd.read_csv(cnt_path, sep='\t', index_col=0)

    pos_path = os.path.join(path, 'data/ST-spotfiles', f'{name}_selection.tsv')
    df_pos = pd.read_csv(pos_path, sep='\t')

    lbl_path = os.path.join(path, 'data/ST-pat/lbl', f'{name}_labeled_coordinates.tsv')
    df_lbl = pd.read_csv(lbl_path, sep='\t')
    df_lbl = df_lbl.dropna(axis=0, how='any')
    df_lbl.loc[df_lbl['label'] == 'undetermined', 'label'] = np.nan
    df_lbl['x'] = (df_lbl['x']+0.5).astype(np.int64)
    df_lbl['y'] = (df_lbl['y']+0.5).astype(np.int64)

    x = df_pos['x'].values
    y = df_pos['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i])) 
    df_pos['id'] = ids

    x = df_lbl['x'].values
    y = df_lbl['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i])) 
    df_lbl['id'] = ids

    meta_pos = df_cnt.join(df_pos.set_index('id'))
    meta_lbl = df_cnt.join(df_lbl.set_index('id'))

    adata = anndata.AnnData(df_cnt, dtype=np.int64)
    adata.obsm['spatial'] = np.floor(meta_pos[['pixel_x','pixel_y']].values).astype(int)
    adata.obs['label'] = pd.Categorical(meta_lbl['label']).codes
    
    img_path = os.path.join(path, 'data/ST-imgs', name[0], name)
    full_image = cv2.imread(os.path.join(img_path, os.listdir(img_path)[0]))
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    patches = []
    for x, y in adata.obsm['spatial']:
        patches.append(full_image[y-size:y+size, x-size:x+size])
    patches = np.array(patches)
    
    return adata, patches

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
    sum=0
    for i in range(t1.shape[0]):
        sum+=(t1[i]-t2[i])**2
    return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n=X.shape[0]
    adj=np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j]=euclid_dist(X[i], X[j])
    return adj

def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    #x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        #beta to control the range of neighbourhood when calculate grey vale for one spot
        #alpha to control the color scale
        beta_half=round(beta/2)
        g=[]
        for i in range(len(x_pixel)):
            max_x=image.shape[0]
            max_y=image.shape[1]
            nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
            g.append(np.mean(np.mean(nbs,axis=0),axis=0))
        c0, c1, c2=[], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0=np.array(c0)
        c1=np.array(c1)
        c2=np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
        c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
        c4=(c3-np.mean(c3))/np.std(c3)
        z_scale=np.max([np.std(x), np.std(y)])*alpha
        z=c4*z_scale
        z=z.tolist()
        print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))
        X=np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X=np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)


def refine(sample_id, pred, dis, num_nbs):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred

def get_predicted_results(dataset, name, path, z):

    if dataset=="DLPFC":
        adata = load_ST_file(os.path.join(path, name))
        df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
        label = pd.Categorical(df_meta['ground_truth']).codes
        n_clusters = label.max() + 1
        adata = adata[label != -1]

        adj_2d = calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].tolist(),
                                      histology=False)

        raw_preds = eval_mclust_ari(label[label != -1], z, n_clusters)

        if len(adata.obs)> 1000:
            num_nbs = 24
        else:
            num_nbs = 4

        refined_preds = refine(sample_id=adata.obs.index.tolist(), pred=raw_preds, dis=adj_2d, num_nbs=num_nbs)
        ari = adjusted_rand_score(label[label != -1], refined_preds)

        return ari, refined_preds

    elif dataset=="Her2st":
        adata, _ = build_her2st_data(path, name)
        label = adata.obs['label']
        n_clusters = label.max() + 1
        adata = adata[label != -1]

        adj_2d = calculate_adj_matrix(x=adata.obsm["spatial"][:, 0].tolist(), y=adata.obsm["spatial"][:, 1].tolist(),
                                      histology=False)

        raw_preds = eval_mclust_ari(label[label != -1], z, n_clusters)

        if len(adata.obs) > 1000:
            num_nbs = 24
        else:
            num_nbs = 4

        refined_preds = refine(sample_id=adata.obs.index.tolist(), pred=raw_preds, dis=adj_2d, num_nbs=num_nbs)
        ari = adjusted_rand_score(label[label != -1], refined_preds)

        return ari, refined_preds
    
    elif dataset=="IDC":
        adata = load_ST_file(os.path.join(path, name))
        df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
        # label = pd.Categorical(df_meta['ground_truth']).codes
        label = pd.Categorical(df_meta['annot_type']).codes
        n_clusters = label.max() + 1
        adata = adata[label != -1]

        adj_2d = calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].tolist(),
                                      histology=False)

        raw_preds = eval_mclust_ari(label[label != -1], z, n_clusters)

        if len(adata.obs)> 1000:
            num_nbs = 24
        else:
            num_nbs = 4

        refined_preds = refine(sample_id=adata.obs.index.tolist(), pred=raw_preds, dis=adj_2d, num_nbs=num_nbs)
        ari = adjusted_rand_score(label[label != -1], refined_preds)

        return ari, refined_preds
