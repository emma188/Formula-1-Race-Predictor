import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from dateutil.relativedelta import *

# query API
from selenium.webdriver.common.by import By

# scoring function for classification
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def pre_process():
    final_df = pd.read_csv('finalcsv.csv',index_col=0)
    df_dum = pd.get_dummies(final_df, columns=['circuit_id', 'nationality', 'constructor'])

    for col in df_dum.columns:
        if 'nationality' in col and df_dum[col].sum() < 140:
            df_dum.drop(col, axis=1, inplace=True)

        elif 'constructor' in col and df_dum[col].sum() < 140:
            df_dum.drop(col, axis=1, inplace=True)

        elif 'circuit_id' in col and df_dum[col].sum() < 70:
            df_dum.drop(col, axis=1, inplace=True)

        else:
            pass

    df = df_dum.copy()
    df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)
    return df

def getTrainTest(df, year1,year2):
    scaler = StandardScaler()
    # training
    train_df = df[(df.season < year1) & (df.season > year2)]
    #train_df.to_csv('test.csv', encoding='utf-8')

    #train_df = df[df.season >= (year-10)]
    train_x = train_df.drop(['driver', 'podium'], axis=1)
    train_y = train_df.podium

    # Standardization

    temp = scaler.fit_transform(train_x)
    train_x = pd.DataFrame(temp, columns=train_x.columns)

    return train_x, train_y

if __name__ == '__main__':
    df = pre_process()
    getTrainTest(df,1985,1983)