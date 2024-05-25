import multiprocessing
n_jobs = multiprocessing.cpu_count()-10

import gc
#import cupy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
from collections import Counter, defaultdict
import scipy
import pickle

from lightgbm import LGBMClassifier
import xgboost

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report, accuracy_score, f1_score
from sklearn import preprocessing
from sklearn.preprocessing import scale, StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, GroupKFold, StratifiedKFold

PATH_FEATURES = '/home/pereligin/host_prediction/Features/'
PATH_SAMPLE_IDS = '/home/pereligin/host_prediction/sample_ids/'
PATH_DATA = '/home/pereligin/host_prediction/data/'
PATH_REPORTS = "/home/pereligin/host_prediction/reports/"
PATH_CLASSIFICATORS = "/home/pereligin/host_prediction/classificators/"

scoring="f1_weighted"

meta_df=pd.read_csv(PATH_DATA+'ranged_genus/data_table_400.tsv', sep=',', index_col = 0)

#meta_df = pickle.load(open(PATH_DATA+'ranged_genus/meta_df_undersampled.pkl', "rb"))
train_ids, test_ids = pickle.load(open(PATH_SAMPLE_IDS+"ranged_genus/train_val_test_400_it0.pkl", "rb"))[0], \
                        pickle.load(open(PATH_SAMPLE_IDS+"ranged_genus/train_val_test_400_it0.pkl", "rb"))[2]

#future_sample_id_uncrossing = pickle.load(open(PATH_SAMPLE_IDS + 'uncrossing_genus_sample.pkl', 'rb'))
#future_sample_id_rhab = pickle.load(open(PATH_SAMPLE_IDS + 'uncrossing_genus_sample.pkl, 'rb'))
#future_sample_id_crossing = pickle.load(open(PATH_SAMPLE_IDS + 'uncrossing_genus_sample.pkl', 'rb'))

def find_best_model(X, y, clf, param_grid):
    model = GridSearchCV(estimator = clf, param_grid = param_grid, cv=5, verbose=10, scoring=scoring, n_jobs=n_jobs)
    model.fit(X,y)
    gc.collect()
    print('Best weighted f1-score of Grid search: {}'.format(model.best_score_))
    print("Best model parameters of Grid search: {}".format(model.best_params_))
    return model.best_estimator_

def svc_prep(data, y_class_h):
         
    classes, list_y = sorted(data[y_class_h].unique()), []
 
    y = np.array(data[y_class_h])
    pre = preprocessing.LabelEncoder()
    pre.fit(classes)  
    y_int = pre.transform(y)
    
    for i in range(len(data[y_class_h].unique())):
        list_y.append(np.array(y_int == i).astype(int))
    
    return(list_y, classes, y_int)

# Достают признаки из файлов

def get_data_pd(feature_set):
    X = pd.read_csv(PATH_FEATURES+feature_set[0]+'.csv', index_col=0)
    for feature in feature_set[1:]:
        X = X.join(pd.read_csv(PATH_FEATURES+feature+'.csv', index_col=0))
    return X

def create_feature_set(features, kmer_lists):
    feature_set = [f'{f}_{k}' for i,f in enumerate(features) for k in kmer_lists[i] ]
    return feature_set

def get_X_y(df_table, df_feature, y_class_h, data_type, train_ids = None, test_ids = None, gbac = None):
    
    if data_type == 'ranged': # Работает для последовательностей, разбитых на участик 250, 400, 800
        
        inside_df = pd.DataFrame({"N_index": range(len(df_table)), "host":df_table.host})

        X_train, X_test = np.array(df_feature.loc[train_ids]), np.array(df_feature.loc[test_ids])
        y_train, y_test = np.array(meta_df.loc[train_ids].host), np.array(meta_df.loc[test_ids].host)
        indices_train, indices_test = np.array(inside_df.loc[train_ids].N_index), np.array(inside_df.loc[test_ids].N_index)
        
        print('X_test size:', len(X_test), 'X_train size:', len(X_train), 'y_test size:', len(y_test), 'y_train size:', len(y_train))
        return(X_test, X_train, y_test, y_train, indices_test, indices_train)  
    
    if data_type == 'full': # Работает для полных последовательностей
        
        df_feature['indices'] = range(len(df_feature))

        test_sample = df_feature.loc[gbac]
        y_test = list(df_table.loc[gbac][y_class_h])
        train_sample = df_feature[~df_feature.isin(test_sample)].dropna()
        y_train = list(df_table.loc[list(train_sample.index)][y_class_h])

        indices_test = test_sample['indices'].to_numpy()
        indices_train = df_feature.loc[train_sample.index]['indices'].to_numpy()

        y_test = np.array(y_test)
        y_train = np.array(y_train)
        X_test = test_sample.iloc[:, :-1].to_numpy()
        X_train = train_sample.iloc[:, :-1].to_numpy()

        print('X_test size:', len(X_test), 'X_train size:', len(X_train), 'y_test size:', len(y_test), 'y_train size:', len(y_train))
        return(X_test, X_train, y_test, y_train, indices_test, indices_train)  


# Отвевает за обучение классификаторов. С помощью переменной models можно задать как одну модель, так и несколько сразу
# Сохраняются в словарь в порядке, в котором модели указаны в переменной models

def calc_models(models, X_train, y_train, X_test, y_test, indices_test, indices_train, y_class_h, list_y, classes, y_int):
    
    classificators = []
    svc_classes = []
    cl_reps = [] 
    for el in models:
        
        if el == 'rf':
            
            param_grid = { 
                "n_estimators": [100, 250, 500],
                "max_features": ['sqrt', 'log2'], #'auto', 
                "max_depth": [None, 5, 12], #3, 10,
                "n_jobs": [1],
                "criterion": ['gini', 'entropy']
                
            }


            model = RandomForestClassifier(random_state=42)
            gridCV_model = find_best_model(X_train, y_train, model, param_grid)
            classificators.append(gridCV_model)
            y_pred = gridCV_model.predict(X_test)
            cl_reps.append(classification_report(y_test, y_pred, output_dict = True, zero_division=1))
            print(el, classification_report(y_test, y_pred))

        if el == 'lgbm':
            
            param_grid = { 
                "n_estimators": [250, 500, 1000],
                "num_leaves": [8, 16, 31], #12, 32
                "max_depth": [-1],#,6,12],
                "reg_lambda": [0, 0.1],
                "reg_alpha": [0, 0.1],
                "device": ["cpu"],
                "n_jobs": [1],
                "learning_rate": [0.1, 0.01]
                
            }
                
            
            
            model = LGBMClassifier(random_state=42, verbose=-1)
            gridCV_model = find_best_model(X_train, y_train, model, param_grid)
            classificators.append(gridCV_model)
            y_pred = gridCV_model.predict(X_test)
            cl_reps.append(classification_report(y_test, y_pred, output_dict = True, zero_division=1))
            print(el, classification_report(y_test, y_pred))
            
        if el == 'xgb':
            le = LabelEncoder()
            y_train, y_test = le.fit_transform(y_train), le.fit_transform(y_test)
            
            param_grid = { 
                "max_depth": [6,18],
                "lambda": [0.1, 1, 10],
                "alpha": [0, 0.1], #0.01
                "n_estimators": [50, 250, 500], #100, 
                #"tree_method": ["hist"],
                "device": ["cpu"],
                "n_jobs": [1],
                "min_child_weight": [3, 12] #7
            }
            
            model = xgboost.XGBClassifier(random_state=42, objective='multi:softprob', eval_metric='auc')
            gridCV_model = find_best_model(X_train, y_train, model, param_grid)
            classificators.append(gridCV_model)
            y_pred = gridCV_model.predict(X_test)
            cl_reps.append(classification_report(y_test, y_pred, output_dict = True, target_names = classes, zero_division=1))
            print(el, classification_report(y_test, y_pred, target_names = classes, zero_division=1))
            
        if el == 'svc':

            param_grid = {
                "C": [1, 10, 1000], #100,
                "gamma": [0.1, "scale", "auto"], # 1, 0.0001, 0.001
                "kernel": ['linear','rbf']
                }


            svc_classificators = [] 
            scale = True
            y_proba = np.zeros(shape=indices_test.shape)
            
            for y_class, class_name in tqdm(zip(list_y, classes)):
                y_train = y_class[indices_train]
                y_test = y_class[indices_test]

                gridCV_model = make_pipeline(StandardScaler(),
                                             GridSearchCV(estimator = SVC(probability=True),
                                                          param_grid = param_grid,
                                                          cv=5,
                                                          verbose=10,
                                                          scoring=scoring,
                                                          refit=True,
                                                          n_jobs=n_jobs,
                                                         )
                                            )
                
                gridCV_model.fit(X_train, y_train)    
                svc_classificators.append(gridCV_model)
                y_pred = gridCV_model.predict(X_test)
                y_proba = np.vstack((y_proba, gridCV_model.predict_proba(X_test)[:,1]))
                print(class_name)

                print(classification_report(y_test, y_pred, target_names = ['Others', class_name], zero_division=1))
                svc_classes.append(classification_report(y_test, y_pred, output_dict=True, target_names = ['Others', class_name], zero_division=1))
        
            classificators.append(svc_classificators)
            y_proba = y_proba[1:]
            y_proba = (y_proba/y_proba.sum(axis=0))


            y_pred_all = np.argmax(y_proba, axis=0)
            cl_reps.append(classification_report(y_int[indices_test], y_pred_all, output_dict = True, target_names = classes, zero_division=1))
            print('F1-score (weighted):', round(f1_score(y_int[indices_test], y_pred_all, average='weighted'), 2))
            print('F1-score (macro):', round(f1_score(y_int[indices_test], y_pred_all, average='macro'), 2))
    return(cl_reps, svc_classes, classificators)
    
    
def training(models, features_sets, sample_type, y_class_h, full_or_ranged, subseq_len = '', best = None):
    
    # Изменение путей исключительно внутри функции. Подготовка индексов для SVC
    
    fun_PATH_FEATURES = PATH_FEATURES
    fun_PATH_CLASSIFICATORS = PATH_CLASSIFICATORS
    fun_PATH_REPORTS = PATH_REPORTS
    BEST_NAME = ''

    
    if full_or_ranged == 'full':
        fun_PATH_FEATURES = PATH_FEATURES[:-1]
        list_y, classes, y_int = svc_prep(meta_df, y_class_h)
        fun_all_sets_names = all_sets_names
        fun_sole_sets_names = sole_sets_names
        
    if full_or_ranged == 'ranged':
        fun_PATH_FEATURES = PATH_FEATURES
        #meta_df_ranged = pd.read_csv(PATH_DATA+'data_table_' + subseq_len + '.tsv', sep=',', index_col = 0)
        #meta_df = pickle.load(open(PATH_DATA+'ranged_genus/meta_df_undersampled.pkl', "rb"))
        #train_ids, test_ids = pickle.load(open(PATH_SAMPLE_IDS+"ranged_genus/train_val_test_ids.pkl", "rb"))[0], \
                        #pickle.load(open(PATH_SAMPLE_IDS+"ranged_genus/train_val_test_ids.pkl", "rb"))[2]
        
        list_y, classes, y_int = svc_prep(meta_df, y_class_h)
        fun_all_sets_names = all_sets_names
        fun_sole_sets_names = sole_sets_names
      
    if best:
        fun_PATH_REPORTS = PATH_REPORTS+'best/'
        fun_PATH_CLASSIFICATORS = PATH_CLASSIFICATORS+'best/'
        fun_all_sets_names = best_sets_all_names
        fun_sole_sets_names = best_sets_sole_names
        BEST_NAME = '_best'
    
    
    all_classificators = {}
    all_clfs = {}
    svc_classes_clfs = {}
    
    # Выбираем список refseq id для тестовой выборки
    
    #if sample_type == 'rhab':
        #future_sample_id = future_sample_id_rhab
    #if sample_type == 'crossing':
        #future_sample_id = future_sample_id_crossing
    #if sample_type == 'uncrossing':
        #future_sample_id = future_sample_id_uncrossing
    
    if len(features_sets[0]) > 1: # если длина 1 призака > 1, то это all_sets (комбинации признаков)
        
        for fset in features_sets: # берём комбинацию признаков (fset)
            
            print("ALL_FEATURES (not sole)")
            
            print(fset, 'Collecting data...')
            X_test, X_train, y_test, y_train, indices_test, indices_train = get_X_y(meta_df, pd.read_csv(fun_PATH_FEATURES+subseq_len+'/'+fset[0]+'.csv', index_col=0), y_class_h, full_or_ranged, train_ids, test_ids)
            y_test, y_train = y_test.astype('str'), y_train.astype('str') 

            for feature in fset[1:]: # далее добавляем остальные признаки из комбинации (fset[1:])
                feature_df = pd.read_csv(fun_PATH_FEATURES+subseq_len+'/'+feature+'.csv', index_col=0)
                X_test_0, X_train_0, y_test_0, y_train_0, indices_test_0, indices_train_0 = get_X_y(meta_df, feature_df, y_class_h, full_or_ranged, train_ids, test_ids)
                print('X_test length: '+ str(len(X_test_0)))
                X_test, X_train = np.hstack((X_test, X_test_0)), np.hstack((X_train, X_train_0))

            print('Data obtained. Calculating models')
            out = calc_models(models, X_train, y_train, X_test, y_test, indices_test, indices_train, y_class_h, list_y, classes, y_int) # Обучаем модели
            all_clfs[fun_all_sets_names[features_sets.index(fset)]], \
            svc_classes_clfs[fun_all_sets_names[features_sets.index(fset)]], \
            all_classificators[fun_all_sets_names[features_sets.index(fset)]] = out[0], out[1], out[2]
            gc.collect()
            
            
        pickle.dump(all_clfs, open(fun_PATH_REPORTS + full_or_ranged + '/' + sample_type + '/' + 'undersampled_400_7-20_f/clfs_all_'+subseq_len+'_'+sample_type+BEST_NAME+'_part2.pkl', 'wb'))
        pickle.dump(svc_classes_clfs, open(fun_PATH_REPORTS + full_or_ranged + '/' + sample_type + '/' + 'undersampled_400_7-20_f/clfs_svc_classes_all_'+subseq_len+'_'+sample_type+BEST_NAME+'_part2.pkl', 'wb'))
        pickle.dump(all_classificators, open(fun_PATH_CLASSIFICATORS + full_or_ranged + '/' + sample_type + '/' + 'undersampled_400_7-20_f/classificators_all_'+subseq_len+'_'+sample_type+BEST_NAME+'_part2.pkl', 'wb'))
    
    if len(features_sets[0]) == 1: # если длина первого призака = 1, то это sole_sets (одиночные признаки)
        
        for fset in features_sets:
    
            print(fset, 'Collecting data...')
            X_test, X_train, y_test, y_train, indices_test, indices_train = get_X_y(meta_df, pd.read_csv(fun_PATH_FEATURES+subseq_len+'/'+fset[0]+'.csv', index_col=0), y_class_h, full_or_ranged, train_ids, test_ids) # Создаём X и y

            print('Data obtained. Calculating models')
            out = calc_models(models, X_train, y_train, X_test, y_test, indices_test, indices_train, y_class_h, list_y, classes, y_int) # Обучаем модели
            all_clfs[fun_sole_sets_names[features_sets.index(fset)]], \
            svc_classes_clfs[fun_sole_sets_names[features_sets.index(fset)]], \
            all_classificators[fun_sole_sets_names[features_sets.index(fset)]] = out[0], out[1], out[2]
            print('gc collect ', len(gc.get_objects()))
            
        pickle.dump(all_clfs, open(fun_PATH_REPORTS + full_or_ranged +'/' + sample_type + '/' + 'undersampled_400_7-20_f/clfs_sole_'+subseq_len+'_'+sample_type+BEST_NAME+'_RNA_7.pkl', 'wb'))
        pickle.dump(svc_classes_clfs, open(fun_PATH_REPORTS + full_or_ranged + '/' + sample_type + '/' + 'undersampled_400_7-20_f/clfs_svc_classes_sole_'+subseq_len+'_'+sample_type+BEST_NAME+'_RNA_7.pkl', 'wb'))
        pickle.dump(all_classificators, open(fun_PATH_CLASSIFICATORS + full_or_ranged + '/' + sample_type + '/' + 'undersampled_400_7-20_f/classificators_sole_'+subseq_len+'_'+sample_type+BEST_NAME+'_RNA_7.pkl', 'wb'))
        
    return
        
# Features
all_sets = [#create_feature_set(['AA'], [list(range(1,3,1))]),
    #create_feature_set(['AA'], [list(range(1,4,1))]),
    #create_feature_set(['AA'], [list(range(1,5,1))]),
    create_feature_set(['DNA'], [list(range(1,3,1))]),
    create_feature_set(['DNA'], [list(range(1,4,1))]),
    create_feature_set(['DNA'], [list(range(1,5,1))]),
    create_feature_set(['DNA'], [list(range(1,6,1))]),
    create_feature_set(['DNA'], [list(range(1,7,1))]),
    create_feature_set(['DNA'], [list(range(1,8,1))]),
    #create_feature_set(['AA'], [[1,3]]),
    create_feature_set(['DNA'], [[1,3,5]]),
    create_feature_set(['DNA'], [[3,5]]),
    create_feature_set(['DNA'], [[2,4,6]]),
    create_feature_set(['DNA'], [[3,5,7]])
    #create_feature_set(['DNA','AA'], [list(range(1,4,1)),list(range(1,3,1))]),
    #create_feature_set(['DNA','AA'], [list(range(2,5,1)),list(range(1,3,1))])
    ]

all_sets_names = [#'AA1-2',
                 #'AA1-3',
                 'RNA1-2', 
                 'RNA1-3', 
                 'RNA1-4', 
                 'RNA1-5', 
                 'RNA1-6', 
                 'RNA1-7',
                 #'AA1,3',
                 'RNA1,3,5',
                 'RNA3,5',
                 'RNA2,4,6',
                 'RNA3,5,7',
                 #'RNA1-3,AA1-2',
                 #'RNA2-4,AA1-2'
                ]

sole_sets = [['DNA_1'],
             ['DNA_2'],
             ['DNA_3'],
             ['DNA_4'],
             ['DNA_5'],
             ['DNA_6'],
             ['DNA_7'],
             #['AA_1'],
             #['AA_2'],
             #['AA_3']
            ]

sole_sets_names = ['RNA_1',
                  'RNA_2',
                  'RNA_3',
                  'RNA_4',
                  'RNA_5',
                  'RNA_6',
                  'RNA_7',
                  #'AA_1',
                  #'AA_2',
                  #'AA_3'
                  ]

                        

                
models = ['rf', 'lgbm', 'xgb', 'svc']

# Пример запуска. Для фрагментов разных длин
# Переменная "models" - список алгоритмов ML, которые будут обучаться
# Переменные "sole_sets" и "all_sets" - набор признаков
# Переменная "sample_type" - тип разбиения ("Непересекающиеся рода вирусов")
# Переменная "full_or_ranged" - признаки фрагментов или геномов используются ('ranged' для фрагментов)
# Переменная "subseq_len" - длина фрагмента

training(models, all_sets, 'uncrossing', 'host', 'ranged', '400')
