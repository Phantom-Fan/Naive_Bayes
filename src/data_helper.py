import csv
import numpy as np
import pandas as pd 

feature_header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

def read_data(train_path, test_path):
    train_data = pd.read_csv(train_path, names=feature_header)
    test_data = pd.read_csv(test_path, names=feature_header)
    return train_data, test_data

def cut_into_bins(train_data, test_data, bin_num=10):
    bins_dict = {}
    for cf in continuous_features:
        labels = [cf + '_bin_' + str(i+1) for i in range(bin_num)]
        train_data[cf], bins = pd.cut(train_data[cf], bin_num, labels=labels, retbins=True)
        bins_dict[cf] = bins
    for cf, bins in bins_dict.items():
        labels = [cf + '_bin_' + str(i+1) for i in range(bin_num)]
        test_data[cf] = pd.cut(test_data[cf], bins, labels=labels)
    return train_data, test_data, bins_dict

if __name__ == '__main__':
    train_data_path = '../data/adult.train'
    test_data_path = '../data/adult.test'
    train_data, test_data = read_data(train_data_path, test_data_path)
    train_data, test_data, bins_dict = cut_into_bins(train_data, test_data)
    print(bins_dict)
    print(train_data.describe())
    print(test_data.describe())