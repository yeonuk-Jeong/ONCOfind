import multiprocessing as mp
from scipy.stats import ks_2samp
import pandas as pd
import gmt_modules as gm
#from sklearn.manifold import TSNE
#import umap
import matplotlib.pyplot as plt
import numpy as np
#import plotly.express as px


def GetES(Value, GeneSet):
    setIDX = list(set(GeneSet).intersection(Value.index))
    
    ks_G = ks_2samp(Value, Value[setIDX], alternative='greater')
    ks_L = ks_2samp(Value, Value[setIDX], alternative='less')
    
    if ks_G[1] < ks_L[1]: ## compare pvalue
        return ks_G[0]
    else:
        return ks_L[0] * -1
    
def GES(ky, Exp, gmt_):
    return [GetES(Exp.iloc[:, i], gmt_[ky]) for i in range(Exp.shape[1])]

def GetES_DF(Exp, gmtFile, core = 1):
    Exp.index = [x.split(':')[0] for x in Exp.index]
    gmt_ = gm.LoadGMT2Dict(gmtFile)
    if Exp.shape[1] > 1000:
        SEQ = list(range(0, Exp.shape[1], 1000))
        SEQ = SEQ + [max(SEQ)+Exp.shape[1] % 1000]
        SpDflist = []
        for i in range(len(SEQ)-1):
            print(i)
            Exp_s = Exp.iloc[:,SEQ[i]:SEQ[i+1]]
            with mp.Pool(core) as pool:
                x = pool.starmap(GES, [(ky, Exp_s, gmt_) for ky in gmt_.keys()])
            SpDflist.append(pd.DataFrame(x, index=gmt_.keys(), columns=Exp_s.columns))
        return pd.concat(SpDflist, axis = 1)
    else:    
        with mp.Pool(core) as pool:
            x = pool.starmap(GES, [(ky, Exp, gmt_) for ky in gmt_.keys()])
        return pd.DataFrame(x, index=gmt_.keys(), columns=Exp.columns)

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

def DimRedc(DF, n_nhb = 11):
    reducer = umap.UMAP(n_neighbors=n_nhb, min_dist=0.4)
    emb_umap = reducer.fit_transform(DF.T)
    tsne = TSNE(n_components=2)
    emb_tsne = tsne.fit_transform(DF.T)
    return {'UMAP':emb_umap, 'TSNE':emb_tsne}

def Emb2Dplot(emb, labels, outfile = './test.jpg', alp = 0.5):
    # 그룹별 색상 설정
    unique_labels = list(set(labels)); unique_labels.sort()
    num_labels = len(unique_labels)
    color_map = plt.cm.get_cmap('tab20b')  # 원하는 컬러맵 선택, 'tab10'은 예시입니다
    
    fig, ax = plt.subplots(figsize=(10, 6))  # 가로 10, 세로 6 비율로 설정
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[color_map(i / num_labels)], label=label, alpha = alp)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()  # 간격 조정
    fig.savefig(outfile, dpi = 100)
    
def Emb2Dplot_html(emb, meta, title = 'no title', outfile = './test.html', alp = 0.8):
    #emb = dr_ocp['UMAP']
    #meta = D_ocp['meta']
    
    df_ = pd.DataFrame(emb, columns=['Dim 1', 'Dim 2'])
    
    df_ = pd.concat([df_, meta.reset_index().iloc[:,1:]], axis=1)
    
    fig = px.scatter(df_, x="Dim 1", y="Dim 2", color="OC_Tissue", symbol='OC_Cancer', 
                 color_discrete_sequence=px.colors.qualitative.Alphabet, opacity = alp)

    fig.update_layout(
        title=title,
        xaxis=dict(linecolor="black"),
        yaxis=dict(linecolor="black"),
        plot_bgcolor="white",
    )

    fig.update_traces(mode="markers")
    fig.write_html(outfile)