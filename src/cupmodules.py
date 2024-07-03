import os, sys, datetime, re
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"]="0" # 첫 번째 GPU를 사용

def mkmodel_rf(mtx, y, savedir):
    os.makedirs('{}/RF'.format(savedir), exist_ok=True)
    
    labelencoder = LabelEncoder()
    y_class_onehot = labelencoder.fit_transform(y)
    y_class_label = keras.utils.to_categorical(y_class_onehot, len(set(y_class_onehot)))
    
    for y_class in set(y_class_onehot):
        TargetCancer = labelencoder.classes_[y_class]
        y_class_now = np.where(y_class_onehot == y_class, 1, 0)
        y_class_label = keras.utils.to_categorical(y_class_now, 2)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=3492)
        rf = RandomForestClassifier(n_estimators=5)  # ,class_weight=class_weight)#"balanced")#{0: 0.8, 1: 1})
        #kfold = KFold(n_splits=5, shuffle=True, random_state=3492)
        #rf = RandomForestClassifier(n_estimators=100)  # ,class_weight=class_weight)#"balanced")#{0: 0.8, 1: 1})
        
        result = cross_val_predict(rf, mtx, y_class_now, cv=kfold)

        auc = metrics.roc_auc_score(y_class_now, result, average='macro', sample_weight=None)
        with open('{}/results.auc.txt'.format(savedir), 'a') as fl:
            fl.write('{}/{} AUC: {} | Case Ratio: {}/{}\n'.format('RF', TargetCancer, np.round(auc, 4), sum(y_class_now), len(y_class_now)))

        rf.fit(mtx, y_class_now)
        saveModel = '{}/RF/model_RF_{}.pickle'.format(savedir, TargetCancer)
        pickle.dump(rf, open(saveModel, 'wb'))
        
def mkmodel_gbm(mtx, y, savedir):
    os.makedirs('{}/GBM'.format(savedir), exist_ok=True)
    
    labelencoder = LabelEncoder()
    y_class_onehot = labelencoder.fit_transform(y)
    y_class_label = keras.utils.to_categorical(y_class_onehot, len(set(y_class_onehot)))
    
    for y_class in set(y_class_onehot):
        TargetCancer = labelencoder.classes_[y_class]
        y_class_now = np.where(y_class_onehot == y_class, 1, 0)
        y_class_label = keras.utils.to_categorical(y_class_now, 2)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=3492)
        gb_clf = GradientBoostingClassifier(
            loss='deviance', ## ‘deviance’, ‘exponential’
            criterion='squared_error', ## 개별 트리의 불순도 측도
            n_estimators=5, ## 반복수 또는 base_estimator 개수
            min_samples_leaf=5, ## 개별 트리 최소 끝마디 샘플 수
            max_depth=3, ## 개별트리 최대 깊이
            learning_rate=0.5, ## 스텝 사이즈
            random_state=100
        )
        
        result = cross_val_predict(gb_clf, mtx, y_class_now, cv=kfold)

        auc = metrics.roc_auc_score(y_class_now, result, average='macro', sample_weight=None)
        with open('{}/results.auc.txt'.format(savedir), 'a') as fl:
            fl.write('{}/{} AUC: {} | Case Ratio: {}/{}\n'.format('GBM', TargetCancer, np.round(auc, 4), sum(y_class_now), len(y_class_now)))

        gb_clf.fit(mtx, y_class_now)
        saveModel = '{}/GBM/model_GBM_{}.pickle'.format(savedir, TargetCancer)
        pickle.dump(gb_clf, open(saveModel, 'wb'))
        del gb_clf
        
def mkmodel_AB(mtx, y, savedir):
    os.makedirs('{}/AB'.format(savedir), exist_ok=True)
    
    labelencoder = LabelEncoder()
    y_class_onehot = labelencoder.fit_transform(y)
    y_class_label = keras.utils.to_categorical(y_class_onehot, len(set(y_class_onehot)))
    
    for y_class in set(y_class_onehot):
        TargetCancer = labelencoder.classes_[y_class]
        y_class_now = np.where(y_class_onehot == y_class, 1, 0)
        y_class_label = keras.utils.to_categorical(y_class_now, 2)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=3492)
        ab_clf = AdaBoostClassifier(n_estimators = 5)
        
        result = cross_val_predict(ab_clf, mtx, y_class_now, cv=kfold)

        auc = metrics.roc_auc_score(y_class_now, result, average='macro', sample_weight=None)
        with open('{}/results.auc.txt'.format(savedir), 'a') as fl:
            fl.write('{}/{} AUC: {} | Case Ratio: {}/{}\n'.format('AB', TargetCancer, np.round(auc, 4), sum(y_class_now), len(y_class_now)))

        ab_clf.fit(mtx, y_class_now)
        saveModel = '{}/AB/model_AB_{}.pickle'.format(savedir, TargetCancer)
        pickle.dump(ab_clf, open(saveModel, 'wb'))
        del ab_clf
        
def mkmodel_DNN(mtx, y, savedir):
    os.makedirs('{}/DNN'.format(savedir), exist_ok=True)
    
    def dnnbin():
        model = Sequential()
        model.add(Dense(100, input_dim=mtx.shape[1], activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
        return model
    
    labelencoder = LabelEncoder()
    y_class_onehot = labelencoder.fit_transform(y)
    y_class_label = keras.utils.to_categorical(y_class_onehot, len(set(y_class_onehot)))
    
    for y_class in set(y_class_onehot):
        TargetCancer = labelencoder.classes_[y_class]
        y_class_now = np.where(y_class_onehot == y_class, 1, 0)
        y_class_label = keras.utils.to_categorical(y_class_now, 2)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=3492)
        dnn_clf = KerasClassifier(build_fn=dnnbin, epochs=100, batch_size=5120, verbose=0)
        
        result = cross_val_predict(dnn_clf, mtx, y_class_now, cv=kfold)

        auc = metrics.roc_auc_score(y_class_now, result, average='macro', sample_weight=None)
        with open('{}/results.auc.txt'.format(savedir), 'a') as fl:
            fl.write('{}/{} AUC: {} | Case Ratio: {}/{}\n'.format('DNN', TargetCancer, np.round(auc, 4), sum(y_class_now), len(y_class_now)))

        dnn_clf.fit(mtx, y_class_now)
        
        json_model = dnn_clf.model.to_json()
        open('{}/DNN/model_DNN_{}.json'.format(savedir, TargetCancer), 'w').write(json_model)
        dnn_clf.model.save_weights('{}/DNN/model_DNN_{}.h5'.format(savedir, TargetCancer), overwrite=True)
       
        del dnn_clf

        
def mkmodel_Logi(mtx, y, savedir):
    os.makedirs('{}/Logi'.format(savedir), exist_ok=True)
    
    labelencoder = LabelEncoder()
    y_class_onehot = labelencoder.fit_transform(y)
    y_class_label = keras.utils.to_categorical(y_class_onehot, len(set(y_class_onehot)))
    
    for y_class in set(y_class_onehot):
        TargetCancer = labelencoder.classes_[y_class]
        y_class_now = np.where(y_class_onehot == y_class, 1, 0)
        y_class_label = keras.utils.to_categorical(y_class_now, 2)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=3492)
        
        lg_clf = LogisticRegression(max_iter=200)
        
        result = cross_val_predict(lg_clf, mtx, y_class_now, cv=kfold)

        auc = metrics.roc_auc_score(y_class_now, result, average='macro', sample_weight=None)
        with open('{}/results.auc.txt'.format(savedir), 'a') as fl:
            fl.write('{}/{} AUC: {} | Case Ratio: {}/{}\n'.format('Logi', TargetCancer, np.round(auc, 4), sum(y_class_now), len(y_class_now)))

        lg_clf.fit(mtx, y_class_now)
        saveModel = '{}/Logi/model_Logi_{}.pickle'.format(savedir, TargetCancer)
        pickle.dump(lg_clf, open(saveModel, 'wb'))
        del lg_clf