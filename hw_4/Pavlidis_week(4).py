import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prediction_model_with_pca(x_train, x_test, y_train, y_test):
    pca = PCA(.75)
    pca.fit(x_train)
    print("pca.n_components = ", pca.n_components_)
    x_train, x_test = pca.transform(x_train), pca.transform(x_test)
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)
    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='')
    ax.set(yticks=[0, 2], xticks=[0, 1])
    plt.show()


def prediction_model_without_pca(x_train, x_test, y_train, y_test):
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)
    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='')
    ax.set(yticks=[0, 2], xticks=[0, 1])
    plt.show()


def main():
    breast_dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(breast_dataset.data, breast_dataset.target,
                                                        stratify=breast_dataset.target, test_size=0.15, random_state=0)
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(x_train)
    # Apply transform to both the training set and the test set.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    prediction_model_with_pca(x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy())
    prediction_model_without_pca(x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy())


if __name__ == '__main__':
    main()
