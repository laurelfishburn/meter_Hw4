#LF 3.4.25

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score

from src.ANN_datapreprocessing import ANNDataPreprocessing

class ANNModelBuilder(ANNDataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ANNModelBuilder, self).__init__(*args, **kwargs)

    def ann(self, X_train, X_test, y_train, y_test):
        #create ANN model
        ANN_classfier = MLPClassifier()

        #train model
        ANN_classfier.fit(X_train, y_train)

        #test model
        ANN_predicted = ANN_classfier.predict(X_test)

        error = 0 
        for i in range(len(y_test)):
            error += np.sum(ANN_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, ANN_predicted)

        return ANN_classfier
