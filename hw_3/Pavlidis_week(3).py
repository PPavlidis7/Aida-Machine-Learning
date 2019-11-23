import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FILE_NAME = 'wdbc.data'
# column names source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/download
# wdbc.data file has not column labels. In order to make easier the coding procedure we add them
COLUMN_NAMES = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


def create_plots(statistics_data):
    objects = ('Mean', 'Standard deviation')
    y_pos = np.arange(2)
    values = [statistics_data['mean'], statistics_data['standard_deviation']]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.show()


def calculate_statistics(file_data):
    statistics_data = {'mean': statistics.mean(file_data['diagnosis']),
                       'media': statistics.median(file_data['diagnosis']),
                       'low_median': statistics.median_low(file_data['diagnosis']),
                       'high_median': statistics.median_high(file_data['diagnosis']),
                       'standard_deviation': statistics.stdev(file_data['diagnosis']),
                       'sample_variance': statistics.variance(file_data['diagnosis'])}
    data_to_print = ''.join('{}: {}\n'.format(key, value) for key, value in statistics_data.items())
    create_plots(statistics_data)
    print(data_to_print)


def create_prediction_model(data):
    x_train, x_test, y_train, y_test = train_test_split(data.drop(['diagnosis', 'id'], axis=1),
                                                        data['diagnosis'], test_size=0.25)
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)
    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('R squared: ', metrics.r2_score(y_test, y_pred))

    # get Confusion Matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='')
    ax.set(yticks=[0, 2], xticks=[0, 1])
    plt.show()


def main():
    file_data = pd.read_csv(FILE_NAME, header=None)
    # convert M into 1 and B into 0
    file_data[1] = file_data[1].astype('category').cat.codes
    file_data.columns = COLUMN_NAMES
    calculate_statistics(file_data)
    create_prediction_model(file_data)


if __name__ == '__main__':
    main()
