import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def data_preprocessing(df):
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
        df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    df.hist(figsize=(20, 20))
    plt.plot()
    plt.show()

    # using these plots, we correct the NaN values
    df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
    df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
    df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
    df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
    df['BMI'].fillna(df['BMI'].median(), inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(df.drop(['Outcome'], axis=1), df.Outcome,
                                                        stratify=df.Outcome, test_size=0.2, random_state=0)
    # data normalization
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test):
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
    k_values = [5, 7]
    helper_dict_evaluation = {}

    # in order to keep code and the model analysis simple, we do not plot the confusion matrices
    for k in k_values:
        for dist_metric in distance_metrics:
            knn = KNeighborsClassifier(n_neighbors=k, metric=dist_metric)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            confusion_matrix = np.asarray(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
            true_pos_neg = np.diag(confusion_matrix)
            false_pos_neg = np.diag(np.fliplr(confusion_matrix))
            sum_values = np.sum(false_pos_neg + true_pos_neg)

            # calculate values that we need later for models' evaluation
            e = np.sum(false_pos_neg)/sum_values
            v = e * (1-e)
            n = 86

            # each dict key is in format metric_k e.g euclidean_5, manhattan_5
            dict_key = '{}_{}'.format(dist_metric, k)
            helper_dict_evaluation[dict_key] = {
                'e': e, 'v': v, 'n': n, 'k': k, 'distance_metric': dist_metric, 'accuracy': accuracy
            }
    evaluate_models(helper_dict_evaluation)


def evaluate_models(helper_dict_evaluation):
    already_check_models = set()
    for first_model, first_model_values in helper_dict_evaluation.items():
        for second_model, second_model_values in helper_dict_evaluation.items():
            if first_model == second_model or (tuple(sorted([first_model, second_model])) in already_check_models):
                continue

            already_check_models.add(tuple(sorted([first_model, second_model])))

            print('-'*80)
            print("Model {} has accuracy {} and model {} has {}".format(first_model, first_model_values['accuracy'],
                                                                        second_model, second_model_values['accuracy']))
            print('\n')
            # all models have the same training set and test sets
            p = (abs(first_model_values['e'] - second_model_values['e'])) / \
                (math.sqrt((first_model_values['v'] + second_model_values['v']) / first_model_values['n']))

            if p <= 2:
                print('There is no significant difference in the test set error rate of two supervised learner models '
                      '{} and {} built with the same training data. P = {}'.format(first_model, second_model, p))
            else:
                print('There is significant difference in the test set error rate of two supervised learner models '
                      '{} and {} built with the same training data. P = {}'.format(first_model, second_model, p))


def main():
    df = pd.read_csv('diabetes_data.csv')

    # there are columns with zeros. This values are not valid so we need to handle them properly
    # problematic columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI
    x_train, x_test, y_train, y_test = data_preprocessing(df)
    train_models(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
