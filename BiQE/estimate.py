# coding =utf-8
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import manhattan_distances
import sys


def pearson_corrcoef(x, y):
    return pearsonr(x, y)[0]


def mean_absolute_error(x, y):

    vec = manhattan_distances(np.array(x).reshape(1,-1), np.array(y).reshape(1,-1))
    summation = np.sum(vec)
    mae = summation / len(y)
    return mae


def root_mean_squared_error(x, y):
    mse = mean_squared_error(x, y)
    rmse = np.sqrt(mse)
    return rmse


def read_File(filePath):
    """Every line corresponds to one hter value"""
    fr = open(filePath, 'r')
    lines = fr.readlines()
    dataList = []
    for line in lines:
        line = float(line)
        dataList.append(line)
    return dataList


def read_csv_File(filePath):
    """Every line corresponds to one gold hter value and one predicted hter value separated by Tab"""
    fr = open(filePath, 'r')
    lines = fr.readlines()
    goldList = []
    preList = []
    for line in lines:
        line = line.strip()
        values = map(float, line.split('\t'))
        goldList.append(values[0])
        preList.append(values[1])
    return goldList,preList


if __name__ == '__main__':
    goldList = read_File(sys.argv[1])
    preList = read_File(sys.argv[2])
    # goldList,preList=read_csv_File(sys.argv[1])
    print ("pearson_corrcoef = %.4f" % pearson_corrcoef(goldList, preList))
    print ("MAE: = %.4f" % mean_absolute_error(goldList, preList))
    print ("RMSE: = %.4f" % root_mean_squared_error(goldList, preList))
