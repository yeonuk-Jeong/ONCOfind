import re, os, pickle
from tensorflow.keras.models import model_from_json
import pandas as pd

CancerTypes = ['PANCREATIC.CANCER', 'RECTAL.CANCER', 'ACC', 'LCC', 'BCC', 'HL', 'non.ATC', 'PPGLs',
               'MM', 'PPC', 'cSCC', 'ADC', 'CERVICAL.CANCER', 'WILMS.TUMOR', 'COLON.CANCER', 'GIST', 'BDC',
               'BREAST.CANCER', 'NHL', 'SKIN.MELANOMA', 'SARCOMA', 'PNET', 'SCC', 'BLC', 'HGBT', 'GBM', 'RCC', 'UVEAL.MELANOMA',
               'STOMACH.CANCER', 'PROSTATE.CANCER', 'MCC', 'non.NPC', 'NPC', 'ATC', 'UTERINE.CANCER', 'HBL',
               'SCLC', 'OVARIAN.CANCER', 'EAC', 'LGBT', 'ESCC', 'HCC']

## getModel
def getPklModel(File):
    with open(File, 'rb') as fl:
        mdl = pickle.load(fl)
    return mdl

## getModel_dnn
def getJsonModel(JsonFile):
    mdl = model_from_json(open(JsonFile).read())
    mdl.load_weights(JsonFile.replace('.json', '.h5'))
    return mdl

def getInFile(InFile):
    df = pd.read_csv(InFile)
    mtx = df.loc[:,[x for x in df.columns if re.search('ID_[0-9]+',x)]]
    mtx.index = df.ID_PATIENT
    return mtx



def GetScores_raw(InFile, modelDir = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/model/230615/RAW/'):
    MldTyps = ['AB', 'RF', 'GBM', 'Logi', 'DNN']
    
    InData = getInFile(InFile)
    
    finDic = {}
    
    ### Non DNN
    for mldtyp in MldTyps[:4]:

        mldpth = os.path.join(modelDir, mldtyp)
        CTnames = ['{}_{}'.format(mldtyp, ct) for ct in CancerTypes]
        #CTnames = [x.replace('model_', '').replace('.pickle', '') for x in os.listdir(mldpth) if x.count('pickle') > 0]

        resDic = {}
        for CT in CTnames:
            mldpkl = os.path.join(mldpth, 'model_{}.pickle'.format(CT))
            model = getPklModel(mldpkl)
            resDic[CT] = model.predict_proba(InData)[0:,1]
        
        resDF = pd.DataFrame(resDic)
        resDF.index = InData.index
        finDic[mldtyp] = resDF
    
    ### DNN
    mldtyp = MldTyps[4]
    mldpth = os.path.join(modelDir, mldtyp)
    CTnames = [x.replace('model_', '').replace('.json', '') for x in os.listdir(mldpth) if x.count('json') > 0]
    
    resDic = {}
    for CT in CTnames:
        mldpkl = os.path.join(mldpth, 'model_{}.json'.format(CT))
        model = getJsonModel(mldpkl)
        resDic[CT] = model.predict_proba(InData)[0:,1]
        
    resDF = pd.DataFrame(resDic)
    resDF.index = InData.index
    finDic[mldtyp] = resDF
        
    return finDic


def GetScores_std(InFile, modelDir = '/BiO/home/jkang/WORK/230615_ONCOfindAI/jw_oncofindai/new_cup_jkang/model/230615/STD/'):
    MldTyps = ['AB', 'RF', 'GBM', 'Logi', 'DNN']
    
    InData = getInFile(InFile)
    
    InData[InData.isna()] = 0
    
    finDic = {}
    for mldtyp in MldTyps[:4]:

        mldpth = os.path.join(modelDir, mldtyp)
        CTnames = ['{}_{}'.format(mldtyp, ct) for ct in CancerTypes]
        #CTnames = [x.replace('model_', '').replace('.pickle', '') for x in os.listdir(mldpth) if x.count('pickle') > 0]

        resDic = {}
        for CT in CTnames:
            mldpkl = os.path.join(mldpth, 'model_{}.pickle'.format(CT))
            model = getPklModel(mldpkl)
            resDic[CT] = model.predict_proba(InData)[0:,1]
        
        resDF = pd.DataFrame(resDic)
        resDF.index = InData.index
        finDic[mldtyp] = resDF

    ### DNN
    mldtyp = MldTyps[4]
    mldpth = os.path.join(modelDir, mldtyp)
    CTnames = [x.replace('model_', '').replace('.json', '') for x in os.listdir(mldpth) if x.count('json') > 0]
    
    resDic = {}
    for CT in CTnames:
        mldpkl = os.path.join(mldpth, 'model_{}.json'.format(CT))
        model = getJsonModel(mldpkl)
        resDic[CT] = model.predict_proba(InData)[0:,1]
        
    resDF = pd.DataFrame(resDic)
    resDF.index = InData.index
    finDic[mldtyp] = resDF
    
    return finDic


### 각 모델별 결과 top 3 -> True
def TopNcheck(df, N = 3):
    binDic = {}
    for i in range(df.shape[0]):
        xx1 = df.iloc[i,:].sort_values(ascending = False)
        xx1[:N] = int(1)
        xx1[N:] = int(0)
        binDic[df.index[i]] = xx1
    resDF = pd.DataFrame(binDic).T
    return resDF.loc[df.index,df.columns]