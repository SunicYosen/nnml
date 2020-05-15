#! /usr/bin/python3

import numpy as np

def info_entropy(data_set):
    counts, feactures = np.shape(data_set)

    label_counts = {}

    for feature_vector in data_set:
        current_label = feature_vector[feactures - 1] # Label At last

        if current_label not in label_counts.keys():
            label_counts[current_label] = 0           # Each Label count

        label_counts[current_label] += 1

    entropy = float(0)

    for key in label_counts:
        probability  = float(label_counts[key]) / counts
        entropy     -= probability * np.log2(probability)

    return entropy

def split_dataset(data_set, axis, value): # The sub data set after split
    sub_data_set=[]
    
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            split_feature_vector = feature_vector[:axis]
            split_feature_vector.extend(feature_vector[axis+1:])
            sub_data_set.append(split_feature_vector)

    return sub_data_set


def best_feature_split(data_set):
    counts, nfeactures = np.shape(data_set)
    number_featrues   = nfeactures - 1

    base_entropy    = info_entropy(data_set)
    best_info_gain  = 0
    best_featrue_i  = -1

    for i in range(number_featrues):
        featrue_list  = [example[i] for example in data_set]
        unique_values = set(featrue_list)

        new_entroy   = 0

        for value in unique_values:
            sub_data_set = split_dataset(data_set, i, value)

            probability  = len(sub_data_set) / float(len(data_set))
            new_entroy  += probability * info_entropy(sub_data_set)

        info_gain = base_entropy - new_entroy

        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_featrue_i = i

    return best_featrue_i


def majority_count(label_list):
    label_count = {}

    for label in label_list:
        if label not in label_count.keys():
            label_count[label] = 0

        label_count[label] += 1

    sort_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)

    return sort_label_count[0][0]

def create_tree(data_set, feactures):

    label_list = [example[-1] for example in data_set]

    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]

    if len(data_set[0]) == 1:
        return majority_count(label_list)

    best_feature_i = best_feature_split(data_set)
    best_feature   = feactures[best_feature_i]

    decision_tree  = {best_feature:{}}

    del(feactures[best_feature_i])

    feacture_values = [example[best_feature_i] for example in data_set]

    unique_values   = set(feacture_values)

    for value in unique_values:
        sub_features = feactures[:]
        decision_tree[best_feature][value] = \
            create_tree(split_dataset(data_set, best_feature_i, value), sub_features)

    return decision_tree