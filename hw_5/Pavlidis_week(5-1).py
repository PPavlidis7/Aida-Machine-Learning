import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import tree


FILE_NAME = 'Titanic.xlsx'


def create_tree_plot(clf, x_train):
    fig, ax = plt.subplots(dpi=200)
    tree.plot_tree(clf, feature_names=x_train.values, class_names=list(x_train.columns.values))
    plt.show()


def main():
    df = pd.read_excel(FILE_NAME, sheet_name='Sheet1')
    # remove the first 2 rows with data:
    # C C   C   C
    # I I   I   O
    df.drop([df.index[0], df.index[1]], axis='rows', inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(['Survived'], axis=1), df.Survived,
                                                        stratify=df.Survived, test_size=0.2, random_state=0)

    x_train, x_test = pd.get_dummies(x_train), pd.get_dummies(x_test)

    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(criterion="entropy")
    # Train Decision Tree Classifier
    clf = clf.fit(x_train, y_train)
    create_tree_plot(clf, x_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='')
    ax.set(yticks=[0, 2], xticks=[0, 1])
    plt.show()


if __name__ == '__main__':
    main()
