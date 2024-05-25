import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

import random

random.seed(123)


					
# Функция - металклассификатор LR (логистическая регрессия)					
def LR(best_features, ranged, train_feature_df=None, test_feature_df=None, feature_df=None):
    
    if ranged:
        X_train, X_test = np.array(train_feature_df[best_features]), np.array(test_feature_df[best_features])
        y_train, y_test = np.array(train_feature_df["y_true"]), np.array(test_feature_df["y_true"])
    else:
        X_train, X_test, y_train, y_test = train_test_split(feature_df[best_features], feature_df.iloc[:,-1], test_size=0.48, random_state=42)
    
    clf = LogisticRegression()


    param_grid = {
                    "penalty":["l2"],  #"l1",
                    "C":[1, 10, 100, 1000],
                    "solver":["lbfgs", "liblinear", "newton-cholesky"], #, "newton-cg", "sag", "saga"
                    "class_weight" : ["balanced"],
                    "max_iter": [300]
                 }


    model = find_best_model(X_train, y_train, clf, param_grid)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Weighted F1-score: "+str(f1_score(y_test, y_pred, average='weighted')))
    #print(np.std(X_train, 0)*model.coef_)
    return model, f1_score(y_test, y_pred, average='weighted')


# Функция - металклассификатор RF (случайный лес)
def RF(best_features, ranged, train_feature_df=None, test_feature_df=None, feature_df=None):
    
    if ranged:
        X_train, X_test = np.array(train_feature_df[best_features]), np.array(test_feature_df[best_features])
        y_train, y_test = np.array(train_feature_df["y_true"]), np.array(test_feature_df["y_true"])
    else:
        X_train, X_test, y_train, y_test = train_test_split(feature_df[best_features], feature_df.iloc[:,-1], test_size=0.48, random_state=42)
    
    
    clf = RandomForestClassifier()


    param_grid = { 
                "n_estimators": [100, 250, 500],
                "max_features": ['sqrt', 'log2'], #'auto', 
                "max_depth": [None, 5, 12], #3, 10,
                "n_jobs": [8],
                "criterion": ['gini', 'entropy']
                
            }


    model = find_best_model(X_train, y_train, clf, param_grid)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Weighted F1-score: "+str(f1_score(y_test, y_pred, average='weighted')))
    #print(np.std(X_train, 0)*model.coef_)
    return model, f1_score(y_test, y_pred, average='weighted')


# Функция для подбора параметров моделей
def find_best_model(X, y, clf, param_grid):
    model = GridSearchCV(estimator = clf, param_grid = param_grid, cv=5, verbose=0, scoring="f1_weighted")
    model.fit(X,y)
    print('Best weighted f1-score of Grid search: {}'.format(model.best_score_))
    print("Best model parameters of Grid search: {}".format(model.best_params_))
    return model.best_estimator_
 

 
# Функция для обучения метаклассификатора
def meta_clf(fragment_len, best_features_dict, RF=False):
    
    best_features = best_features_dict[fragment_len]
    if fragment_len == "genome":
        ranged, train_feature_df, test_feature_df = False, None, None
        feature_df = pd.read_csv("prediction_tables/regression_features_full.csv", sep="\t", index_col=0)
        
    
    else:
        ranged, feature_df = True, None    
        train_feature_df = pd.read_csv("prediction_tables/train_prediction_ranged_"+fragment_len+".csv", sep="\t", index_col=0)
        test_feature_df = pd.read_csv("prediction_tables/test_prediction_ranged_"+fragment_len+".csv", sep="\t", index_col=0)
        
        """
        X, X_train, X_test = pd.concat([train_feature_df, test_feature_df]).drop("y_true", axis=1).values, \
                             train_feature_df.drop("y_true", axis=1).values, \
                             test_feature_df.drop("y_true", axis=1).values

        y, y_train, y_test = pd.concat([train_feature_df, test_feature_df])["y_true"].values, \
                             train_feature_df["y_true"].values, \
                             test_feature_df["y_true"].values
        """
        
    out_lr = LR(best_features, ranged, train_feature_df, test_feature_df, feature_df)
    if RF:
        out_rf = RF(best_features, ranged, train_feature_df, test_feature_df, feature_df)
        return out_lr, out_rf
    return out_lr
    
    
    
# Наборы лучших признаков (по результатом бинарной классификации)  
best_features_genome = ["RNA_4_rf_insecta",
                     "RNA_4_rf_mammalia",
                     #"AA_1_rf_viridiplantae",

                     "RNA_4_lgbm_insecta",
                     "RNA1-2_lgbm_mammalia",
                     #"RNA1-3,AA1-2_lgbm_viridiplantae",

                     "RNA_4_xgb_insecta",
                     "RNA_7_xgb_mammalia",
                     "RNA_1_xgb_viridiplantae",

                     "RNA_2_svc_insecta_positive",
                     "RNA3,5,7_svc_mammalia_positive",
                     #"RNA2-4,AA1-2_svc_viridiplantae_positive",
                        ]

best_features_800 = ["RNA1-3_rf_insecta",
                     "RNA_3_rf_mammalia",
                     "RNA_3_rf_viridiplantae",

                     "RNA1-2_lgbm_insecta",
                     "RNA1-4_lgbm_mammalia",
                     "RNA2,4,6_lgbm_viridiplantae",

                     "RNA3,5,7_xgb_insecta",
                     "RNA_2_xgb_mammalia",
                     "RNA_3_xgb_viridiplantae",

                     "RNA_4_svc_insecta_positive",
                     "RNA_3_svc_mammalia_positive",
                     "RNA1-3_svc_viridiplantae_positive",
                    ]

best_features_400 = ["RNA1-6_rf_insecta",
                     "RNA1-4_rf_mammalia",
                     "RNA1-2_rf_viridiplantae",

                     "RNA2,4,6_lgbm_insecta",
                     "RNA1-5_lgbm_mammalia",
                     "RNA1-4_lgbm_viridiplantae",

                     "RNA1-4_xgb_insecta",
                     "RNA3,5,7_xgb_mammalia",
                     "RNA1-3_xgb_viridiplantae",

                     "RNA1-5_svc_insecta_positive",
                     "RNA_4_svc_mammalia_positive",
                     "RNA1-4_svc_viridiplantae_positive",
                    ]

best_features_dict = {"genome": best_features_genome,
                 "800": best_features_800,
                 "400": best_features_400
                }

# Пример использования
# Переменная "fragment_len" - длина используемых фрагментов или полные геномы
# Переменная "best_features_dict" - словарь с наилучшими признаками по результатам бинарной классификации


clf = meta_clf(fragment_len, best_features_dict)[0]                    