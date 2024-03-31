import math

import pandas as pd
import numpy as np
import requests
import torch
from matplotlib import pyplot as plt

# scoring function for classification
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

modell = None
modell_score = 0.0
instanx = instany = None


def score_classification(df, model, year):
    global instanx, instany
    score = 0
    for circuit in df[df.season == year]['round'].unique():
        test = df[(df.season == year) & (df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis=1)
        y_test = test.podium

        instanx = X_test.iloc[1:2]
        instany = pd.DataFrame(columns=X_test.columns)
        # print("X", X_test.iloc[1], "Y", y_test.iloc[1])

        # scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # make predictions
        tmp = model.predict_proba(X_test)
        # print(tmp[1])
        prediction_df = pd.DataFrame(tmp, columns=['proba_0', 'proba_1'])
        prediction_df['actual'] = y_test.reset_index(drop=True)
        prediction_df.sort_values('proba_1', ascending=False, inplace=True)
        prediction_df.reset_index(inplace=True, drop=True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)

        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == year]['round'].unique().max()
    return model_score


# scoring function for regression

def score_regression(df, model, year):
    score = 0
    global instanx
    for circuit in df[df.season == year]['round'].unique():
        test = df[(df.season == year) & (df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis=1)
        y_test = test.podium

        instanx = X_test.iloc[1:7]
        instany = X_test.iloc[2]

        # scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict(X_test), columns=['results'])
        prediction_df['actual'] = y_test.reset_index(drop=True)
        # prediction_df['actual'] = prediction_df.podium.map(lambda x: 1 if x == 1 else 0)
        prediction_df.sort_values('results', ascending=False, inplace=True)
        prediction_df.reset_index(inplace=True, drop=True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)

        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == year]['round'].unique().max()
    return model_score


def read_data():
    final_df = pd.read_csv('finalcsv.csv', index_col=0)
    return final_df


def process_data(final_df):
    # get dummies
    cols = []
    if ('circuit_id' in final_df.columns):
        cols.append('circuit_id')
    if ('nationality' in final_df.columns):
        cols.append('nationality')
    if ('constructor' in final_df.columns):
        cols.append('constructor')
    df_dum = pd.get_dummies(final_df, columns=cols)

    for col in df_dum.columns:
        if 'nationality' in col and df_dum[col].sum() < 140:
            df_dum.drop(col, axis=1, inplace=True)

        elif 'constructor' in col and df_dum[col].sum() < 140:
            df_dum.drop(col, axis=1, inplace=True)

        elif 'circuit_id' in col and df_dum[col].sum() < 70:
            df_dum.drop(col, axis=1, inplace=True)

        else:
            pass

    # print(final_df)
    # final_df.to_csv('finalcsv.csv', encoding='utf-8')

    df = df_dum.copy()
    df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)
    t1 = df.drop(['driver', 'podium'], axis=1)
    t1 = pd.DataFrame(columns=t1.columns)
    return df, t1


# split train
# get training and test set
def getTrainTest(df, year, year2=None):
    # training
    if (year2 is not None):
        train_df = df[(df.season < year) & (df.season >= year2)]
    else:
        train_df = df[(df.season < year)]
    # train_df = df[(df.season < year) & (df.season >= (year-10))]
    train_x = train_df.drop(['driver', 'podium'], axis=1)
    train_y = train_df.podium

    # Standardization

    temp = scaler.fit_transform(train_x)
    train_x = pd.DataFrame(temp, columns=train_x.columns)

    return train_x, train_y


scaler = StandardScaler()
years = [2014, 2015, 2016, 2017, 2018, 2019]

# gridsearch dictionary

comparison_dict = {'model': [],
                   'params': [],
                   'score': []}


# Neural network

def NN(df, hidden_layer_sizes=(80, 20, 40, 5), activation='identity', solver='lbfgs', alpha=0.1082636733874054,
       year=2019, year2=None):
    global modell, modell_score
    if (year2 is None):
        X_train, y_train = getTrainTest(df, year)
    else:
        X_train, y_train = getTrainTest(df, year, year2)

    modell = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                           random_state=1)
    modell.fit(X_train, y_train)

    modell_score = score_classification(df, modell, year)
    return modell, modell_score


def NNReg(df, hidden_layer_sizes=(75, 30, 50, 10, 3), activation='identity', solver='sgd', alpha=0.4832930238571752,
          year=2019, year2=None):
    global modell, modell_score
    if (year2 is None):
        X_train, y_train = getTrainTest(df, year)
    else:
        X_train, y_train = getTrainTest(df, year, year2)

    modell = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                          random_state=1)
    modell.fit(X_train, y_train)

    modell_score = score_regression(df, modell, year)
    return modell, modell_score


def ranFor(df, crit='entropy', maxf='auto', maxd=49.0, year=2019, year2=None):
    # Random Forest Classifier
    global modell, modell_score
    if (year2 is None):
        X_train, y_train = getTrainTest(df, year)
    else:
        X_train, y_train = getTrainTest(df, year, year2)

    modell = RandomForestClassifier(criterion=crit, max_features=maxf, max_depth=maxd)
    modell.fit(X_train, y_train)

    modell_score = score_classification(df, modell, year)
    return modell, modell_score


def svmmod(df, cw={0: 1, 1: 1}, c=10.0, kernel='sigmoid', gamma=0.0001, year=2019, year2=None):
    # Support Vector Machines
    global modell, modell_score
    if (year2 is None):
        X_train, y_train = getTrainTest(df, year)
    else:
        X_train, y_train = getTrainTest(df, year, year2)

    modell = svm.SVC(probability=True, class_weight=cw, C=c, kernel=kernel, gamma=gamma)
    modell.fit(X_train, y_train)

    modell_score = score_classification(df, modell, year)
    return modell, modell_score


def svmreg(df, cw={0: 1, 1: 1}, c=10.0, kernel='linear', gamma=0.0001, year=2019, year2=None):
    # Support Vector Machines
    global modell, modell_score
    if (year2 is None):
        X_train, y_train = getTrainTest(df, year)
    else:
        X_train, y_train = getTrainTest(df, year, year2)

    modell = svm.SVR(C=c, kernel=kernel, gamma=gamma)
    modell.fit(X_train, y_train)

    modell_score = score_regression(df, modell, year)
    return modell, modell_score


def linreg(df, year=2019, year2=None):
    # Linear Regressor
    global modell, modell_score
    if (year2 is None):
        X_train, y_train = getTrainTest(df, year)
    else:
        X_train, y_train = getTrainTest(df, year, year2)

    modell = LinearRegression()
    modell.fit(X_train, y_train)

    modell_score = score_regression(df, modell, year)
    return modell, modell_score


def featImp(df):
    # Random Forest Classifier

    X_train, y_train = getTrainTest(df, 2018)
    feature_names = X_train.columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    # forest_importances = forest_importances.iloc[:15]

    for1 = []
    for i in forest_importances.index:
        if ('nationality' in i):
            for1.append(forest_importances[i])
            forest_importances.drop(labels=i, inplace=True)
    forest_importances['nationality'] = np.mean(for1)
    for1 = []
    for i in forest_importances.index:
        if ('constructor' in i):
            if (('pos' not in i) and ('points' not in i) and ('wins' not in i)):
                for1.append(forest_importances[i])
                forest_importances.drop(labels=i, inplace=True)
    forest_importances['constructor'] = np.mean(for1)
    for1 = []
    for i in forest_importances.index:
        if ('circuit_id' in i):
            for1.append(forest_importances[i])
            forest_importances.drop(labels=i, inplace=True)
    forest_importances['circuit'] = np.mean(for1)
    for1 = []
    for i in forest_importances.index:
        if ('weather' in i):
            for1.append(forest_importances[i])
            forest_importances.drop(labels=i, inplace=True)
    forest_importances['weather'] = np.mean(for1)

    forest_importances.sort_values(ascending=False, inplace=True)
    forest_importances.to_csv('featimp.csv', encoding='utf-8')

    fig, ax = plt.subplots()
    forest_importances.plot.bar()
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('featimp593.png')
    plt.show()


def sigmoid(num):
    return 1.0 / (1.0 + math.exp(-num))


def inferencemany(modtype, model, instances):
    if ('classification' in modtype):
        tmp = model.predict_proba(instances)[:, 1]
    else:
        tmp = model.predict(instances)
    tmp = [sigmoid(i) for i in tmp]
    tmp = [float(i) / sum(tmp) for i in tmp]
    return (np.argmax(tmp), np.max(tmp) * 100)


def inferenceone(modtype, model, instance):
    if ('classification' in modtype):
        tmp = model.predict_proba(np.expand_dims(np.array(instance), 0))[:, 1]
    else:
        tmp = model.predict(np.expand_dims(np.array(instance), 0))
    return tmp * 100

# df = read_data()
# df,_ = process_data(df)
# t1,t2 = svmmod(df,year=2019,year2=2011)
# print(instanx,instanx.iloc[0]['weather_warm'])
# instany.loc[0] = 0.0
# instany.loc[0]['round'] = 1.0
# instany.loc[1]['weather_dry'] = 1
# print(instany)
