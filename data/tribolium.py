__author__ = 'panc'

import pandas as pd

from mixedlogistic.preprocess import *


def import_data():
    datafile = "/Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/Paper_mixedLogistic/software/mixedlogistic/data/tribolium.csv"
    return pd.read_csv(datafile)


def prepareTriboliumData():
    """
    Preparing tribolium data for estimation.
    :return:
    """
    # read data
    df = import_data()
    xm = pd.get_dummies(df.Replicate).ix[:, 2:]
    xr = pd.get_dummies(df.Species).ix[:, 1:]
    m = np.matrix(df.Total).T
    y = np.matrix(df.Remaining).T
    # pre-process data
    xm = addIntercept(xm)  # add a leading 1s column
    xr = addIntercept(xr)  # add a leading 1s column
    c = 3  # three replicates as three components
    return {'c':c, 'y':y, 'xm':xm, 'xr':xr, 'm':m}