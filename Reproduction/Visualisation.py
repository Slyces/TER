import tensorflow as tf, numpy as np, pandas

processed = pandas.read_csv("processed.csv", sep=",")

sess = tf.Session()

saver = tf.train.import_meta_graph('my-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

model = tf.get_collection('model')
X, Y, p_keep_input, p_keep_hidden = tf.get_collection('placeholders')

# =============================================================================
# Visualisation
Ps = []
Xs = np.array([
    np.concatenate([processed.loc[i:i + 9, ('Nasdaq', 'Dow',
                                                      'S&P500', 'Rates')[j]]
        for j in range(4)]) for i in range(len(processed.Date)-11)
    ])
Ys = np.array([[processed.loc[i + 10, ('Nasdaq', 'Dow',
                                                'S&P500', 'Rates')[j]]
        for j in range(4)] for i in range(len(processed.Date)-11)
    ])

for i in range(len(processed.Date) - 11):
    Ps += list(*sess.run(model, feed_dict={X: Xs[i].reshape((1,40)), Y: Ys[i].reshape((1,4)),
                                        p_keep_input: 1.0, p_keep_hidden: 1.0}))
Ps = np.array(Ps)

from datetime import datetime

dates = np.array([datetime.strptime(x, '%Y-%m-%d') for x in processed.loc[11:, 'Date']])
# =======================================================================================
from bokeh.plotting import figure, output_file, show
from bokeh.models import DatetimeTickFormatter
from bokeh.models.layouts import Column
from math import pi

output_file("datetime.html")

subs = [None for i in range(4)]
for i in range(4):
    subs[i] = figure(width=800, height=350, x_axis_type="datetime")
    subs[i].line(dates, Ps[:, i], color='navy', alpha=0.8)
    subs[i].line(dates, Ys[:, i], color='red', alpha=0.9)
    subs[i].xaxis.formatter=DatetimeTickFormatter(
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
    subs[i].xaxis.major_label_orientation = pi/4

p = Column(*subs)

show(p)
