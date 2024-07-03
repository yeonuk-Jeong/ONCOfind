import pandas as pd
import numpy as np
import os, sys
sys.path.append('/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/src/')
import copy
import time
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import keras
from sklearn.model_selection import cross_val_predict
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
import matplotlib.pyplot as plt
import ESmodules
import joblib


class OncoFinder:
    def __init__(self):
        self.TestES: pd.DataFrame = None
        self.TestES_ne: pd.DataFrame = None ## normal tissue effect fixed
        self.TestMetaNormTissue: pd.Series = None
        self.DataES: pd.DataFrame = None
        self.DataMeta: pd.DataFrame = None
        self.par: dict = {'pf_auc_cut': 0.5,
                          'multcls_fi_cut': -2.4,
                          'norm_ratio': 0.3,
                          'num_fi': 200, ## max number
                          'n_est': 100}
        self.dir_data_summary = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/DATA_SET_230908/DataSummary_2/DS2_231107_SUMMARY_1_Tissue_Cancer_SRC.csv'
        self.tis_typs: set = None
        self.can_typs: set = None
        #self.cg14835 = pd.read_csv('/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/DATA_SET_230908/CommonGenes_14835.tsv', sep='\t', index_col=0)
        self.cg14835 = pd.read_csv('/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/Modeling_240326_/data/CommonGenes_14875.tsv', sep='\t', index_col=0)
        self.gidtbl = pd.read_csv('/BiO_DB/Reference/human/NCBI/GeneIDmapping/results_220225/220225_ENSG2ENTREZ.tsv', sep='\t', index_col=0)
        self.GMTfile = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/data/MsigDB/c6c8_merged.all.v2023.1.Hs.entrez.gmt'
        self.MultiClsFeatureImportance: pd.DataFrame = None
        self.MODEL_DIR = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/Modeling_231030/rf_model_save_1/'
        
        self.DIR_OCP = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/DATA_SET_230908/DATASET1_231027/'
        self.DIR_GTEX = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/DATA_SET_230908/DATASET1-GTEx_231027/'
        self.DIR_TCGA = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/DATA_SET_230908/DATASET1-TCGA_231106/'
        self.n_j = 100 ## threads
    
    def dataload(self, Dir):
        tis_list = pd.Series(os.listdir(Dir))[pd.Series(os.listdir(Dir)).isin(self.tis_typs)]

        es_ = []
        meta_ = []

        for tis in tis_list:
            Dir_t = os.path.join(Dir, tis)

            Dir_t_e = os.path.join(Dir_t, [x for x in os.listdir(Dir_t) if x.count('_exp.c6c8.tsv')][0])
            Dir_t_m = os.path.join(Dir_t, [x for x in os.listdir(Dir_t) if x.count('_meta.tsv')][0])

            e_ = pd.read_csv(Dir_t_e, sep='\t', index_col=0)
            m_ = pd.read_csv(Dir_t_m, sep='\t', index_col=0)

            es_.append(e_)
            meta_.append(m_)

        es_df = pd.concat(es_, axis = 1)
        meta_df = pd.concat(meta_, axis = 0)

        return {'es':es_df, 'meta':meta_df}
    
    def getDATA(self):
        
        ClsInfo = pd.read_csv(self.dir_data_summary)
        self.tis_typs = set(ClsInfo['OC_Tissue'])
        self.can_typs = set(ClsInfo['OC_Cancer'])

        D_ocp = self.dataload(self.DIR_OCP)

        # D_ocp 중복제거
        D_ocp['meta'] = D_ocp['meta'].loc[~D_ocp['meta'].index.duplicated(), :] 
        D_ocp['es'] = D_ocp['es'].loc[:, ~D_ocp['es'].columns.duplicated()] 
        #print(sum(D_ocp['meta'].index != D_ocp['exp'].columns))

        D_tcga = self.dataload(self.DIR_TCGA)
        D_gtex = self.dataload(self.DIR_GTEX)

        ## remove tcga metastatsis
        D_tcga['meta'] = D_tcga['meta'][[False if x.endswith('m') else True for x in D_tcga['meta'].OC_Cancer]]
        D_tcga['es'] = D_tcga['es'].loc[:,D_tcga['meta'].index]

        ### merge all data
        mrg_df_es = pd.concat([D_ocp['es'], D_tcga['es'], D_gtex['es']], axis = 1)

        A1 = D_ocp['meta'].loc[:,['OC_Tissue', 'OC_Cancer']]
        A1.loc[:,'OC_SRC'] = 'OCP'

        A2 = D_tcga['meta'].loc[:,['OC_Tissue', 'OC_Cancer']]
        A2.loc[:,'OC_SRC'] = 'TCGA'

        A3 = D_gtex['meta'].loc[:,['OC_Tissue', 'OC_Cancer']]
        A3.loc[:,'OC_SRC'] = 'GTEx'

        mrg_df_meta = pd.concat([A1, A2, A3], axis = 0)

        self.DataES = mrg_df_es
        self.DataMeta = mrg_df_meta

    ## MultiClass FI
    def MultiRF_FI(self):

        DataMeta_TO = self.DataMeta[self.DataMeta.OC_SRC.isin(['TCGA', 'OCP'])]
        DataMeta_TO[[True if x.count('Normal') == 0 else False for x in DataMeta_TO.OC_Cancer]]
        DataES_TO = self.DataES.loc[:,DataMeta_TO.index]

        y = DataMeta_TO.OC_Tissue
        labelencoder = LabelEncoder()
        y_class_onehot = labelencoder.fit_transform(y)
        y_class_label = keras.utils.to_categorical(y_class_onehot, len(set(y_class_onehot)))

        rf_model = RandomForestClassifier(n_estimators=self.par['n_est'], random_state=42, n_jobs = self.n_j)
        rf_model.fit(DataES_TO.T, y_class_label)

        return [pd.DataFrame({'fea_importance':np.log10(rf_model.feature_importances_)}, index=rf_model.feature_names_in_), rf_model]
    
    ## Array(OCP) vs RSEQ(TCGA) batch effect features remove
    def BE_auc(self):
        aucRes = {}
        for ix in self.DataES.index:
            e_o = self.DataES.loc[ix,self.DataMeta[(self.DataMeta.OC_SRC == 'OCP') & (self.DataMeta.OC_Cancer != 'Normal')].index]
            e_t = self.DataES.loc[ix,self.DataMeta[(self.DataMeta.OC_SRC == 'TCGA') & (self.DataMeta.OC_Cancer != 'Normal')].index]

            a = [0]*len(e_o) + [1]*len(e_t); b = list(e_o) + list(e_t)

            metrics.roc_auc_score(a, b)

            aucRes[ix] = metrics.roc_auc_score(a, b)

        return pd.DataFrame([aucRes], index=['AUC']).T
    
    def mkFItbl(self):
        f1 = self.MultiRF_FI()[0]
        f2 = self.BE_auc()
        self.MultiClsFeatureImportance = pd.concat([f1,f2], axis = 1)
        
    def loadTPM2ES(self, tpmfl = '/BiO_NGS/NGS/RSEQ_Storage/RESULTS/byProject_GRCh38/HN00179097.TPM.mat.txt', sep = '\t', ENSG = True):
        
        if ENSG: ## STAR/RSEM TPM 
            ex_ = pd.read_csv(tpmfl, sep=sep, index_col=0, comment = '#')
            cidx = list(set(ex_.index).intersection(set(self.gidtbl.index)))
            ex_ = ex_.loc[cidx,]
            gidtbl_f = self.gidtbl.loc[cidx,:]
            gidtbl_f = gidtbl_f.loc[~gidtbl_f.index.duplicated(),:]
            ex_.loc[:,'Entrez'] = [str(x) for x in gidtbl_f.loc[cidx,'Entrez']]
            ex_ = ex_.groupby('Entrez').sum()
        else: ## array upc(entrez id)
            ex_ = pd.read_csv(tpmfl, sep=sep, index_col=0, comment = '#')
            ex_.index = [str(x) for x in ex_.index]
        
        cgGs = set([x.split(':')[0] for x in self.cg14835['GeneID']])
        exGs = set(ex_.index)
        CGS = list(cgGs.intersection(exGs))
        ex_ = ex_.loc[CGS,:]
        
        '''
        ex_ = ex_[ex_.index.isin([x.split(':')[0] for x in self.cg14835['GeneID']])]
        ex_ = ex_.loc[[x.split(':')[0] for x in self.cg14835['GeneID']],:]
        ex_.index = self.cg14835['GeneID']
        '''
        
        es_ = ESmodules.GetES_DF(ex_, gmtFile=self.GMTfile, core = self.n_j)
        self.TestES = es_
        return es_
        
    def extNormTis(self, es_i, NormTissue = 'Ovary', normRatio = 0.3):
        if NormTissue in set(self.DataMeta[self.DataMeta.OC_Cancer == 'Normal'].OC_Tissue):
            DataMeta_N = self.DataMeta[self.DataMeta.OC_Tissue.isin([NormTissue])]
            DataMeta_N = DataMeta_N[DataMeta_N.OC_Cancer == 'Normal']
            DataES_N = self.DataES.loc[es_i.index,DataMeta_N.index]
            return ((es_i.T - (normRatio * DataES_N.mean(axis = 1)))/(1-normRatio)).T
        else:
            return es_i

    def extNormTis_DF(self):
        ESdf_e = copy.deepcopy(self.TestES)
        for ID in ESdf_e.columns:
            ESdf_e.loc[:,ID] = self.extNormTis(es_i = ESdf_e.loc[:,ID], NormTissue = self.TestMetaNormTissue[ID], normRatio = self.par['norm_ratio'])
        self.TestES_ne = ESdf_e
        
    def TRAIN(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        DataMeta_TO = self.DataMeta[self.DataMeta.OC_SRC.isin(['TCGA', 'OCP'])]
        DataMeta_TO[[True if x.count('Normal') == 0 else False for x in DataMeta_TO.OC_Cancer]]
        
        
        a_h = 0.5 + self.par['pf_auc_cut']; a_l = 0.5 - self.par['pf_auc_cut']
        fi_c = self.par['multcls_fi_cut']

        feat_selc = self.MultiClsFeatureImportance[(self.MultiClsFeatureImportance.fea_importance > fi_c) & 
                                                   (self.MultiClsFeatureImportance.AUC > a_l) & 
                                                   (self.MultiClsFeatureImportance.AUC < a_h)].index
        
        DataES_TO = self.DataES.loc[feat_selc,DataMeta_TO.index]

        for Tis in list(set(DataMeta_TO.OC_Tissue)):

            y = [1 if x == Tis else 0 for x in DataMeta_TO.OC_Tissue]

            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(DataES_TO.T, y)
            gb_model = RandomForestClassifier(random_state=42, n_jobs = self.n_j, n_estimators=self.par['n_est'])
            gb_model.fit(X_resampled, y_resampled)

            ## N features
            n_feat = pd.DataFrame({'feat':gb_model.feature_names_in_, 'fi':gb_model.feature_importances_}).sort_values('fi', ascending = False).reset_index().iloc[:self.par['num_fi'],:].feat
            del gb_model

            ## feature cut
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(DataES_TO.loc[n_feat,:].T, y)
            gb_model = RandomForestClassifier(random_state=42, n_jobs = self.n_j, n_estimators=self.par['n_est'])
            gb_model.fit(X_resampled, y_resampled)
            AUC_ = metrics.roc_auc_score(y_resampled, gb_model.predict(X_resampled))
            joblib.dump(gb_model, os.path.join(self.MODEL_DIR, 
                                               '{}_a{}_f{}_n{}_i{}_e{}_auc{}.joblib'.format(Tis, 
                                                                                            self.par['pf_auc_cut'],
                                                                                            self.par['multcls_fi_cut'],
                                                                                            self.par['norm_ratio'],
                                                                                            self.par['num_fi'],
                                                                                            self.par['n_est'], 
                                                                                            AUC_))) ## auc/ficut/norm/finum/est
            
            del gb_model    
            
    def PRED(self):

        DataMeta_TO = self.DataMeta[self.DataMeta.OC_SRC.isin(['TCGA', 'OCP'])]
        DataMeta_TO[[True if x.count('Normal') == 0 else False for x in DataMeta_TO.OC_Cancer]]
        
        res = []
        
        for Tis in list(set(DataMeta_TO.OC_Tissue)):
            trgfn = '{}_a{}_f{}_n{}_i{}_e{}'.format(Tis, 
                                                    self.par['pf_auc_cut'],
                                                    self.par['multcls_fi_cut'],
                                                    self.par['norm_ratio'],
                                                    self.par['num_fi'],
                                                    self.par['n_est']) ## auc/ficut/norm/finum/est
            
            gb_model = joblib.load([os.path.join(self.MODEL_DIR, x) for x in os.listdir(self.MODEL_DIR) if x.count(trgfn) > 0][0])
            
            res.append(pd.DataFrame(gb_model.predict_proba(self.TestES_ne.loc[gb_model.feature_names_in_,:].T), columns=['p1',Tis])[Tis])
            del gb_model
        
        resDF = pd.DataFrame(res)#.sort_values(0, ascending=False)
        resDF.columns = self.TestES_ne.columns
        
        return resDF