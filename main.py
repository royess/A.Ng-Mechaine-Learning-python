import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def plot_data():
    # load data
    def load_data(direct):
        with open(direct, 'r', newline='') as f:
            data = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            return [row for row in data]
            # can be traversed for only one time

    temp = np.asarray(load_data('ex2data1.txt'))
    X = temp[..., 0:2]
    y = temp[..., 2]

    # plot
    df_pos = pd.DataFrame([X[i, ...] for i in range(len(X)) if y[i] == 1],
                          columns=['Exam1', 'Exam2'])
    df_neg = pd.DataFrame([X[i, ...] for i in range(len(X)) if y[i] == 0],
                          columns=['Exam1', 'Exam2'])

    ax = df_pos.plot.scatter(x='Exam1', y='Exam2', color='DarkBlue', label='Admitted')
    df_neg.plot.scatter(x='Exam1', y='Exam2', color='DarkGreen', label='Not Admitted', ax=ax)
    plt.show()

if __name__ == '__main__':
    plot_data()
