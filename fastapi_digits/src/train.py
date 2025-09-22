'''
from sklearn.tree import DecisionTreeClassifier
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    joblib.dump(dt_classifier, "../model/iris_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
'''
from sklearn.ensemble import RandomForestClassifier
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a RandomForest Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=12)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "../model/digits_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)