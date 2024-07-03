import pandas as pd
import numpy as np
import gmt_modules as gm
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from sklearn.manifold import TSNE
import umap
from scipy import stats
import ESmodules as es

def DimRedc(DF):
    reducer = umap.UMAP(n_neighbors=11, min_dist=0.4)
    emb_umap = reducer.fit_transform(DF.T)
    tsne = TSNE(n_components=2)
    emb_tsne = tsne.fit_transform(DF.T)
    return {'UMAP':emb_umap, 'TSNE':emb_tsne}

def Emb2Dplot(emb, labels, outfile = './test.jpg'):
    # 그룹별 색상 설정
    unique_labels = list(set(labels)); unique_labels.sort()
    num_labels = len(unique_labels)
    color_map = plt.cm.get_cmap('tab20b')  # 원하는 컬러맵 선택, 'tab10'은 예시입니다
    
    fig, ax = plt.subplots(figsize=(10, 6))  # 가로 10, 세로 6 비율로 설정
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[color_map(i / num_labels)], label=label, alpha = 0.5)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()  # 간격 조정
    fig.savefig(outfile, dpi = 100)


def rmDupIdx(df):
    # 중복되는 인덱스 확인
    duplicate_indexes = list(set(df.index[df.index.duplicated(keep='last')]))
    non_duplicate_indexes = list(set(df.index) - set(duplicate_indexes))
    

    df_max_list = []
    for duplicate_index in duplicate_indexes:
        df_max_list.append(df.loc[duplicate_index,:].max())
        
    nmaxdf = pd.concat(df_max_list, axis=1)
    nmaxdf.columns = duplicate_indexes
    nmaxdf = nmaxdf.T
    
    return pd.concat([df.loc[non_duplicate_indexes,:], nmaxdf], axis=0)

meta = pd.read_csv('../Modeling_230721/DATA/data_meta.tsv', sep ='\t', index_col = 0)
exp = pd.read_csv('../Modeling_230721/DATA/data_exp.tsv', sep ='\t', index_col = 0)

meta_tcga = pd.read_csv('../Modeling_230721/TCGA/tcga_meta.tsv', sep ='\t', index_col = 0)
exp_tcga = pd.read_csv('../Modeling_230721/TCGA/tcga_fpkm_entrez.tsv', sep ='\t', index_col = 0)

IDX = list(set(exp_tcga.index).intersection(set(exp.index)))

exp_ = exp.loc[IDX,:]

exp_tcga_ = rmDupIdx(exp_tcga.loc[IDX,:]).loc[IDX,:]

print(exp_tcga_.shape)
print(exp_.shape)

es.GetES_DF(exp_, 'c6.all.v2023.1.Hs.entrez.gmt').to_csv('ES_OCP_ALL_C6.csv')
es.GetES_DF(exp_tcga_, 'c6.all.v2023.1.Hs.entrez.gmt').to_csv('ES_TCGA_ALL_C6.csv')
es.GetES_DF(np.log2(exp_tcga_+1), 'c6.all.v2023.1.Hs.entrez.gmt').to_csv('ES_TCGA_log2_ALL_C6.csv')

es.GetES_DF(exp_, 'c2.cp.kegg.v2023.1.Hs.entrez.gmt').to_csv('ES_OCP_ALL_C2.csv')
es.GetES_DF(exp_tcga_, 'c2.cp.kegg.v2023.1.Hs.entrez.gmt').to_csv('ES_TCGA_ALL_C2.csv')
es.GetES_DF(np.log2(exp_tcga_+1), 'c2.cp.kegg.v2023.1.Hs.entrez.gmt').to_csv('ES_TCGA_log2_ALL_C2.csv')

es.GetES_DF(exp_, 'c8.all.v2023.1.Hs.entrez.gmt').to_csv('ES_OCP_ALL_C8.csv')
es.GetES_DF(exp_tcga_, 'c8.all.v2023.1.Hs.entrez.gmt').to_csv('ES_TCGA_ALL_C8.csv')
es.GetES_DF(np.log2(exp_tcga_+1), 'c8.all.v2023.1.Hs.entrez.gmt').to_csv('ES_TCGA_log2_ALL_C8.csv')