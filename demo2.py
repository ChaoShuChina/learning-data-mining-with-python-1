__author__ = 'chao-shu'
# OneR
from  sklearn.datasets import load_iris
import numpy as np
from collections import defaultdict
from operator import itemgetter
from sklearn.cross_validation import train_test_split


def train_feature_value(X, y_ture, feature_index, value):
    class_count = defaultdict(int)
    for sample, y in zip(X, y_ture):
        if sample[feature_index] == value:
            class_count[y] += 1
    sorted_class_counts = sorted(class_count.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    incorrect_predictions = [class_count for class_value, class_count in class_count.items()
                             if class_value != most_frequent_class]
    error = sum(incorrect_predictions)
    return most_frequent_class, error


def train_on_feature(X, y_true, feature_index):
    values = set(X[:, feature_index])
    predictors = {}
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature_index, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error


dataset = load_iris()
X = dataset.data
Y = dataset.target
attribute_means = X.mean(axis=0)
X_d = np.array(X >= attribute_means, dtype='int')
Xd_train, Xd_test, Y_train, Y_test = train_test_split(X_d, Y, random_state=14)
all_predictors = {}
errors = {}
for feature_index in range(Xd_train.shape[1]):
    predictors, total_error = train_on_feature(Xd_train, Y_train, feature_index)
    all_predictors[feature_index] = predictors
    errors[feature_index] = total_error
best_feature, best_error = sorted(errors.items(), key=itemgetter(1))[0]
model = {'variable': best_feature,
         'predictor': all_predictors[best_feature][0]}


# variable = model['variable']
# predictor = model['predictor']
# prediction = predictor[int(sample[variable])]
def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted


y_predicted = predict(Xd_test, model)
accuracy = np.mean(y_predicted == Y_test) * 100
print(accuracy)
# print(dataset.DESCR)
