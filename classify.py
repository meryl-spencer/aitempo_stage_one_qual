import numpy as np
import pandas as pd
import matplotlib; matplotlib.rcParams.update({'font.size': 14})
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
import argparse
import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', type=str, required=True, dest='training')
    parser.add_argument('-te', '--test', type=str, required=True, dest='testing')
    args = parser.parse_args()

    training_data = pd.read_csv(args.training)
    testing_data = pd.read_csv(args.testing)

    #Do some normilization to strech all the data 0-1
    for feat in training_data.columns:
        if not feat in ['age', 'pid']:
            minval = np.nanmin(training_data[feat].values)
            maxval = np.nanmax(training_data[feat].values)
            norm_train = (np.array(training_data[feat].values)-minval)/(maxval-minval)
            training_data[feat] = norm_train

            norm_test = (np.array(testing_data[feat].values) - minval) / (maxval - minval)
            testing_data[feat] = norm_test

    X_train = np.array(training_data)[:, 2:]
    y_train = np.array(training_data['age'].values)

    X_test = np.array(testing_data)[:, 2:]

    pid_test = testing_data['pid'].values

    print('Training a gradient boost classifier ...')
    classifier = HistGradientBoostingClassifier(max_iter=10).fit(X_train, y_train)
    y_predict_train = classifier.predict(X_train)
    print('Gradient Boost precision on training data', sklearn.metrics.precision_score(y_train, y_predict_train, average='macro'))
    print('Gradient Boost recall on training data', sklearn.metrics.recall_score(y_train, y_predict_train, average='macro'))

    y_predict_test = classifier.predict(X_test)

    prediction_dict = {}
    for pid, predicted_class in zip(pid_test, y_predict_test):
        prediction_dict[int(pid)] = predicted_class

    with open("aitempo_output.json", 'w') as f:
        f.write(pprint.pformat(prediction_dict))

if __name__ == "__main__":
    main()
