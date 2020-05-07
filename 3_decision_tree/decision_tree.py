import operator
import pandas as pd
import numpy as np

OutlookIndex = ["Sunny", "Overcast", "Rain"]
HumidityIndex = ["Normal", "High"]
WindyIndex = ["Weak", "Strong"]


def util_cal_entropy(X):
    num_entries = len(X)
    label_counts = {}
    for entry in X:
        current_label = entry[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        entropy -= prob * np.log2(prob)
    return entropy


def majority_count(label_list):
    label_count = {}
    for entry in label_list:
        if entry not in label_count.keys():
            label_count[entry] = 0
        label_count[entry] += 1

    sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]


def split_dataset(dataset, feature, value):
    ret_dataset = []
    for entry in dataset:
        if entry[feature] == value:
            reduced_feat_vec = list(entry[:feature])
            reduced_feat_vec.extend(entry[feature + 1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_best_feature(dataset):
    feature_num = len(dataset[0]) - 1
    base_entropy = util_cal_entropy(dataset)
    best_information_gain = 0.0
    best_feature = -1
    for i in range(feature_num):
        feature_list = [entry[i] for entry in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0
        for val in unique_vals:
            sub_dataset = split_dataset(dataset, i, val)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * np.log2(prob)
        information_gain = base_entropy - new_entropy

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_feature = i
    return best_feature


def decision_tree_build(dataset, y):
    label_list = [row[-1] for row in dataset]
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    if len(dataset[0]) == 1:
        return majority_count(label_list)

    best_feature = choose_best_feature(dataset)
    best_feature_label = y[best_feature]
    decision_tree = {best_feature_label: {}}
    del (y[best_feature])
    feat_values = [entry[best_feature] for entry in dataset]
    unique_vals = set(feat_values)
    for val in unique_vals:
        sub_labels = y[:]
        decision_tree[best_feature_label][val] = decision_tree_build(split_dataset(dataset, best_feature, val),
                                                                     sub_labels)
    return decision_tree


def main():
    # dataset = np.genfromtxt(fname="WeatherConditions.csv", delimiter=",")
    # X, y = np.split(dataset, [-1], axis=1)
    # dataset = np.concatenate((X, y), axis=1)
    dataset = pd.read_csv('WeatherConditions.csv', delimiter=',')
    y = list(dataset.columns.values)
    dataset = dataset.values
    tree = decision_tree_build(dataset, y)
    print(tree)


if __name__ == '__main__':
    main()
