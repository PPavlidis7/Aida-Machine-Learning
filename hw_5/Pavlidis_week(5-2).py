import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import tree


FILE_NAME = 'Cardiology.xlsx'


def create_tree_plot(clf, x_train):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=400)
    tree.plot_tree(clf, feature_names=x_train.values, class_names=list(x_train.columns.values))
    plt.savefig('Cardiology_tree.png')


def main():
    df = pd.read_excel(FILE_NAME, sheet_name='Sheet1')
    x_train, x_test, y_train, y_test = train_test_split(df.drop(['class'], axis=1), df['class'],
                                                        stratify=df['class'], test_size=0.25, random_state=0)

    x_train, x_test = pd.get_dummies(x_train), pd.get_dummies(x_test)

    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(criterion="entropy")
    # Train Decision Tree Classifier
    clf = clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='')
    ax.set(yticks=[0, 2], xticks=[0, 1])
    plt.show()

    create_tree_plot(clf, x_train)


if __name__ == '__main__':
    main()
