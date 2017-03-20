import numpy as np
import pandas as pd 
import data_helper
import math

def get_posterier(frequency_dict, d_less, d_greater):
    '''
        input the frequency dict of the train set
        return a dict
        posterier[feature_name][label] = p(xi|label)
        Use Laplacian correction
    '''
    posterier = {}
    for name, feature_value in frequency_dict.items():
        N_i = len(feature_value) # number of possible values for feature_name
        for value, counts in feature_value.items():
            posterier[value] = {}
            posterier[value]['<=50K.'] = (counts[0] + 1) / (d_less + N_i)
            posterier[value]['>50K.'] = (counts[1] + 1) / (d_greater + N_i)
    return posterier

def sample_dataset(train_set, frac=1.):
    if frac < 1.:
        small = train_set.sample(frac=frac, replace=False)
        return small
    else:
        return train_set

def get_log_likelihood(features, label, prior, posterier):
    log_likelihood = 0.0
    for feature in features:
        if isinstance(feature, str) and feature != '?':
            log_likelihood += np.log(posterier[feature][label])
    log_likelihood += np.log(prior[label])
    return log_likelihood

def predict_sample(feature, label, prior, posterier):
    less_log_prob = get_log_likelihood(feature, '<=50K.', prior, posterier)
    greater_log_prob = get_log_likelihood(feature, '>50K.', prior, posterier)
    prediction = '<=50K.' if less_log_prob > greater_log_prob else '>50K.'
    if prediction == '<=50K.' and label == '<=50K.':
        return 'tp'
    elif prediction == '<=50K.' and label == '>50K.':
        return 'fp'
    elif prediction == '>50K.' and label == '<=50K.':
        return 'fn'
    elif prediction == '>50K.' and label == '>50K.':
        return 'tn'

def train(train_data, frac=1.):
    feature_names = train_data.columns 

    '''
        init a frequency dict to hold the statistics for train set
        frequency_dict[feature_name][feature_value] = [value_for_<50, value_for_>50]
    '''
    frequency_dict = {}
    for name in feature_names[0:-1]:
        frequency_dict[name] = {}
        for value in train_data[name].unique():
            if value != '?':
                frequency_dict[name][value] = [0., 0.]

    # sample the train set
    train_set = sample_dataset(train_data, frac)

    # get the distribution of labels. Use Laplacian correction
    label_counts = train_set['label'].value_counts().values
    prior = {}
    prior['<=50K.'] = (label_counts[0] + 1) / (label_counts[0] + label_counts[1] + 2)
    prior['>50K.'] = (label_counts[1] + 1) / (label_counts[0] + label_counts[1] + 2)

    # get the frequency
    for name in feature_names[0:-1]:
        grouped_dataframe = train_set.groupby([name, 'label']).count().fillna(0)
        indices = grouped_dataframe.index
        pair_list = [pair for pair in indices]
        count_list = grouped_dataframe.values[:, 0]
        for pair, count in zip(pair_list, count_list):
            if pair[0] != '?':
                fill_idx = 0 if pair[1] == '<=50K.' else 1
                frequency_dict[name][pair[0]][fill_idx] = float(count)

    posterier = get_posterier(frequency_dict, label_counts[0], label_counts[1])
    return prior, posterier

def evaluate(test_data, prior, posterier):
    results = {'tp':0, 'fp':0, 'tn':0, 'fn':0}
    test_set = test_data.values 
    labels = test_set[:, -1]
    features = test_set[:, 0:-1]
    evaluate_number = test_set.shape[0]
    for i in range(evaluate_number):
        res = predict_sample(features[i], labels[i], prior, posterier)
        results[res] += 1
    acc = float(results['tp'] + results['tn']) / (results['tp'] + results['tn'] + results['fp'] + results['fn'])
    precision = float(results['tp']) / (results['tp'] + results['fp'])
    recall = float(results['tp']) / (results['tp'] + results['fn'])
    f1 = float(2 * results['tp']) / (2 * results['tp'] + results['fp'] + results['fn'])
    return (acc, precision, recall, f1)

if __name__ == '__main__':
    train_data_path = '../data/adult.train'
    test_data_path = '../data/adult.test'
    train_data, test_data = data_helper.read_data(train_data_path, test_data_path)
    train_data, test_data, bins_dict = data_helper.cut_into_bins(train_data, test_data)

    columns_to_del = ['capital-gain', 'capital-loss']
    train_data.drop(columns_to_del, axis=1, inplace=True)
    test_data.drop(columns_to_del, axis=1, inplace=True)
    
    total_result = []
    for i in range(1): 
        prior, posterier = train(train_data)
        print(prior)
        result = evaluate(test_data, prior, posterier)
        total_result.append(list(result))
    total_result = np.asarray(total_result)
    print(np.mean(total_result, axis=0), np.max(total_result, axis=0), np.min(total_result, axis=0))
