import numpy as np
from sklearn.datasets import load_digits
from numpy.random import randint
import numpy.random
import sys




def train_test_split(X, Y):
    # spliting test dataset of size 500 with 50 entries of each class
    r_state= numpy.random.RandomState(13)
    classes = [0]*10
    test_set_idx = []
    test_size = 500
    data_size=X.shape[0]

    while test_size != 0:
        rand_num = r_state.randint(0, data_size-1)
        if rand_num not in test_set_idx:
            if classes[Y[rand_num]] < 50:
                classes[Y[rand_num]] += 1
                test_set_idx.append(rand_num)
                test_size -= 1

    x_test = X[test_set_idx]
    y_test = Y[test_set_idx]
    train_set_idx = []

    for i in range(data_size):
        if i not in test_set_idx:
            train_set_idx.append(i)                                 #creating training dataset

    x_train,y_train = X[train_set_idx],Y[train_set_idx]

    return x_train, y_train, x_test, y_test

def l2_norm(x, y):                  #euclidian l2_norm distance
    distance = np.sqrt(np.square(x-y).sum())
    return distance


def knn(k, x_train, y_train, x_test):
      #knn classifier
    preds = []
    for test_data in x_test:
        distances = []
        for train_data in x_train:
            distances.append(l2_norm(train_data, test_data))
        k_nns = np.argpartition(distances, k)[0:k]
        pred = y_train[k_nns]
        counts = np.bincount(pred)
        non_zero = counts[counts != 0]
        if len(np.unique(non_zero)) == 1:
            output = pred[0]
        else:
            output = np.argmax(counts)
        preds.append(output)
    return preds


def accuracy(predictions, ground_truth):    
    #meaasuring accuracy
    acc_score = sum(predictions == ground_truth) / ground_truth.shape[0]
    return acc_score


if __name__ == '__main__':
    digits = load_digits()
    print("Dataset loaded")
    X,Y = digits.data,digits.target
    x_train, y_train, x_test, y_test = train_test_split(X, Y)
    k_values=[1,3,5,7]
    for k in k_values:
    #knn preditctions for different k values
      print("K value= ",k)
      train_predictions = knn(k, x_train, y_train, x_train)
      test_predictions = knn(k, x_train, y_train, x_test)
      print('\nTrain Accuracy : ',accuracy(train_predictions, y_train), '\n')
      print('Test Accuracy : ',accuracy(test_predictions, y_test), '\n')
      print('---------------------------------------------------\n')
