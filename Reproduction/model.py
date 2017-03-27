# =============================================================================
# Imports
import tensorflow as tf, numpy as np, pandas
# =============================================================================

# =============================================================================
# Création du modèle
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# Notre modèle sera, d'après l'article, sur une architecture 40 - 41 - 41 - 4
# <inputs> - 40
# <hidden> - 41
# <hidden> - 41
# <output> -  4
def model(X, w_h1, w_h2, w_o, p_keep_input=1., p_keep_hidden=1.):
    # =========================================================================
    # X    : les inputs, shape [n x 40]
    # w_h1 : les poids du premier hidden layer, shape [40 x 41]
    # w_h2 : les poids du second hidden layer, shape [41 x 41]
    # w_o  : les poids du layer d'output, shape [41 x 4]
    # vérification : [n x 40] [40 x 41] [41 x 41] [41 x 4] = [n x 4]
    # =========================================================================

    X = tf.nn.dropout(X, p_keep_input)
    # --> dropout désactive certains neuronnes avec une certaine probabilité
    #       pour empêcher l'over-fitting
    h1 = tf.nn.tanh(tf.matmul(X, w_h1))
    # relu = fonction d'activation
    h1 = tf.nn.dropout(h1, p_keep_hidden)

    h2 = tf.nn.tanh(tf.matmul(h1, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h1, w_o)

X = tf.placeholder("float", [None, 40])
Y = tf.placeholder("float", [None, 4])

w_h = init_weights([40, 20])
w_h2 = init_weights([20, 20])
w_o = init_weights([20, 4])

p_keep_input = tf.placeholder("float")   # La probabilité qui peut changer
p_keep_hidden = tf.placeholder("float")  # La probabilité qui peut changer
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden) # On construit
#  le modèle avec X l'input externe

cost = tf.reduce_sum(tf.square(py_x - Y))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# =============================================================================

# =============================================================================
# On charge les données
processed = pandas.read_csv("processed.csv", sep=",")

n = len(processed.Date.values) // 11

# On découpe par tranches de 11
Xdatas = np.array([
    np.concatenate([processed.loc[i * 11:i * 11 + 9, ('Nasdaq', 'Dow',
                                                      'S&P500', 'Rates')[j]]
        for j in range(4)]) for i in range(n)
    ])
Ydatas = np.array([[processed.loc[i * 11 + 10, ('Nasdaq', 'Dow',
                                                'S&P500', 'Rates')[j]]
        for j in range(4)] for i in range(n)
    ])

p_train = 0.85
indexes = np.random.rand(n) < p_train
trX = Xdatas[indexes]
trY = Ydatas[indexes]
teX = Xdatas[~indexes]
teY = Ydatas[~indexes]
# =============================================================================

# =============================================================================
sess = tf.Session()
# Init variables
tf.global_variables_initializer().run(session=sess)

print("Begin of training")
# Training
print(len(trX))
for i in range(4000):
    for start, end in zip(range(0, len(trX), 50), range(50, len(trX)+1, 50)):
        # batches of 10
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                        p_keep_input: 1.0, p_keep_hidden: 1.0})
    if i % 500 == 0:
        print(i, sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],
                                            p_keep_input: 1.0, p_keep_hidden: 1.0}))
# =============================================================================


# =============================================================================
# Visualisation
predicted = [None for i in range(10)]
Ys = []
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

Ps = []
for i in range(len(processed.Date) - 11):
    Ps += list(sess.run(py_x, feed_dict={X: Xs[i].reshape((1,40)), Y: Ys[i].reshape((1,4)),
                                        p_keep_input: 1.0, p_keep_hidden: 1.0}))
Ps = np.array(Ps)
print(Ps.shape, Ps)

from matplotlib import pyplot as plt
from matplotlib import dates as dt
from datetime import datetime

dates = np.array([datetime.strptime(x, '%Y-%m-%d') for x in processed.loc[11:, 'Date']])
dates = dt.date2num(dates)

subs = [None for i in range(4)]
for i in range(4):
    subs[i] = plt.subplot(221 + i)

    subs[i].plot_date(dates[:], Ps[:, i], xdate=True, ls='-', label='Prediction',
                      lw=.4, ms=.1)
    subs[i].plot_date(dates[:], Ys[:, i], xdate=True, ls='-', label='Historic Data',
                      lw=.4, ms=.1)
    subs[i].set_title("Subplot {}".format(i))
    handles, labels = subs[i].get_legend_handles_labels()
    subs[i].legend(handles, labels)
plt.show()