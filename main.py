import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.optimize import minimize


def load_data(direct):
    with open(direct, 'r', newline='') as f:
        data = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        return [row for row in data]
        # can be traversed for only one time


def plot_data(X, y):
    # plot
    df_pos = pd.DataFrame([X[i, ...] for i in range(len(X)) if y[i] == 1],
                          columns=['Exam1', 'Exam2'])
    df_neg = pd.DataFrame([X[i, ...] for i in range(len(X)) if y[i] == 0],
                          columns=['Exam1', 'Exam2'])

    ax = df_pos.plot.scatter(x='Exam1', y='Exam2', color='DarkBlue', label='Admitted')
    df_neg.plot.scatter(x='Exam1', y='Exam2', color='DarkGreen', label='Not Admitted', ax=ax)
    plt.show()


def cost_function(theta, X, y):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    hx = sigmoid(np.dot(X, theta))
    cost = -y*np.log(hx) - (1-y)*np.log(1-hx)
    J = sum(cost)/len(y)
    return J

if __name__ == '__main__':
    # load data
    temp = np.asarray(load_data('ex2data1.txt'))
    X_1 = temp[..., 0:2]
    y_1 = temp[..., 2]

    plot_data(X_1, y_1)

    m, n = X_1.shape
    X_1 = np.column_stack(((np.ones((m, 1)), X_1)))
    theta0 = np.asanyarray([0]*(n+1))

    print('use Nelder-Mead method to minimize cost function')
    minimize(lambda t: cost_function(t, X_1, y_1), theta0,
             method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})

    '''
    minimize(lambda t: cost_function(t, X_1, y_1), theta0,
             method='BFGS', options={'disp': True})
    '''