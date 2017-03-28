import tensorflow as tf, numpy as np, pandas
from datetime import datetime

from bokeh.plotting import figure, output_file, show
from bokeh.models import DatetimeTickFormatter
from bokeh.models.layouts import Column
from math import pi

import os

def load_data(path=""):
    # loading processed datas
    processed = pandas.read_csv(os.path.join(path, "processed.csv"), sep=",")
    # Creating base computational datas and obtained values
    Xs = np.array([
                      np.concatenate([processed.loc[i:i + 9, ('Nasdaq', 'Dow',
                                                              'S&P500', 'Rates')[j]]
                                      for j in range(4)]) for i in range(len(processed.Date) - 11)
                      ])
    Ys = np.array([[processed.loc[i + 10, ('Nasdaq', 'Dow',
                                           'S&P500', 'Rates')[j]]
                    for j in range(4)] for i in range(len(processed.Date) - 11)
                   ])

    dates = np.array([datetime.strptime(x, '%Y-%m-%d') for x in processed.loc[11:, 'Date']])
    return Xs, Ys, dates


def restore_model(path=""):
    sess = tf.Session()

    saver = tf.train.import_meta_graph(os.path.join(path, 'Model/my-model.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(path, 'Model/./')))
    return sess, tf.get_collection('model'), \
           tf.get_collection('placeholders')


def predict(Xs, Ys, Ds, path=""):
    sess, model, (X, Y, p_ki, p_kh) = restore_model(path=path)

    Ps = []
    for i in range(len(Xs)):
        Ps.append(sess.run(model, feed_dict={X: Xs[i].reshape((1, 40)),
                                               Y: Ys[i].reshape((1, 4)),
                                               p_ki: 1.0, p_kh: 1.0})[0][0])

    return np.array(Ps), Ys, Ds


def plot_feedforward(height=350, width=800, path=""):
    Ps, Ys, Ds = predict(*load_data(path), path=path)

    axes = [None for i in range(4)]
    indexes = 'Nasdaq DowJones S&P500 Rates'.split()
    for i in range(4):
        axes[i] = figure(width=width, height=height, x_axis_type="datetime")
        axes[i].line(Ds, Ps[:, i], legend="Predictions", color='navy', alpha=0.8)
        axes[i].line(Ds, Ys[:, i], legend="Historic Data", color='red', alpha=0.9)
        axes[i].xaxis.formatter = DatetimeTickFormatter(
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )
        axes[i].xaxis.major_label_orientation = pi / 4

    return dict([(indexes[i], axes[i]) for i in range(4)])


if __name__ == '__main__':
    output_file("datetime.html")
    p = plot_feedforward()
    show(p)
