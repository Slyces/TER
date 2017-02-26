import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from typing import List, Tuple, NewType, Union

Vector = NewType('Vector', List[Union[int, float]])


dates = []
prices = []

def get_data(filename: str= 'aapl.csv'):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            if row[0].split('-')[1:] == ['Mar','16']:
                dates.append(int(row[0].split('-')[0]))
                prices.append(float(row[1]))
    return

def predict_prices(dates: Vector, prices: Vector, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_len = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3, degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1)
    print('Checkpoint 0')
    svr_len.fit(dates, prices)
    print('Checkpoint 1')
    svr_rbf.fit(dates, prices)
    print('Checkpoint 2')
    # svr_poly.fit(dates, prices)
    print('Checkpoint 3')

    print("I'm here !")

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_len.predict(dates), color='green', label='Linear model')
    # plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return tuple([svr.predict(x)[0] for svr in (svr_len, svr_rbf)])


if __name__ == '__main__':
    get_data('aapl.csv')
    predicted_price = predict_prices(dates, prices, 29)

    print(predicted_price)
