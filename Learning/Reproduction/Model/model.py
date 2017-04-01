# =============================================================================
# Imports
import tensorflow as tf, numpy as np, pandas


# =============================================================================

# =============================================================================
# Création du modèle
def init_weights(shape, name="weights"):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


# Notre modèle sera, d'après l'article, sur une architecture 40 - 41 - 41 - 4
# <inputs> - 40
# <hidden> - 41
# <hidden> - 41
# <output> -  4
def model(X, w_h1, w_h2, w_o, b_h1, b_h2, b_o, p_keep_input=1., p_keep_hidden=1.):
    # =========================================================================
    # X    : les inputs, shape [n x 40]
    # w_h1 : les poids du premier hidden layer, shape [40 x 41]
    # w_h2 : les poids du second hidden layer, shape [41 x 41]
    # w_o  : les poids du layer d'output, shape [41 x 4]
    # vérification : [n x 40] [40 x l1] [l1 x l2] [l2 x 4] = [n x 4]
    # =========================================================================

    X = tf.nn.dropout(X, p_keep_input)
    # --> dropout désactive certains neuronnes avec une certaine probabilité
    #       pour empêcher l'over-fitting
    h1 = tf.nn.tanh(tf.add(tf.matmul(X, w_h1), b_h1))
    # relu = fonction d'activation
    h1 = tf.nn.dropout(h1, p_keep_hidden)

    h2 = tf.nn.tanh(tf.add(tf.matmul(h1, w_h2), b_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.add(tf.matmul(h1, w_o), b_o, name="py_x")


X = tf.placeholder("float", [None, 40], name="X")
Y = tf.placeholder("float", [None, 4], name="Y")

tf.add_to_collection('placeholders', X)
tf.add_to_collection('placeholders', Y)

w_h1 = init_weights([40, 20], name='w_h1')
w_h2 = init_weights([20, 20], name='w_h2')
w_o = init_weights([20, 4], name='w_o')

b_h1 = tf.Variable(tf.zeros([20]), name="b_h1")
b_h2 = tf.Variable(tf.zeros([20]), name="b_h2")
b_o = tf.Variable(tf.zeros([4]), name="b_o")

tf.add_to_collection('vars', w_h1)
tf.add_to_collection('vars', w_h2)
tf.add_to_collection('vars', w_o)

tf.add_to_collection('vars', b_h1)
tf.add_to_collection('vars', b_h2)
tf.add_to_collection('vars', b_o)


p_keep_input = tf.placeholder("float", name='p_keep_input')
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

tf.add_to_collection('placeholders', p_keep_input)
tf.add_to_collection('placeholders', p_keep_hidden)

py_x = model(X, w_h1, w_h2, w_o, b_h1, b_h2, b_o,
             p_keep_input, p_keep_hidden)

tf.add_to_collection('model', py_x)

cost = tf.reduce_sum(tf.square(py_x - Y), name='cost')
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
for i in range(50000 * 20):
    for start, end in zip(range(0, len(trX), 50), range(50, len(trX) + 1, 50)):
        # batches of 10
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                      p_keep_input: 1.0, p_keep_hidden: 1.0})
    if i % 1000 == 0:
        print(i, sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],
                                           p_keep_input: 1.0, p_keep_hidden: 1.0}))
    if i % 50000 == 0:
        saver = tf.train.Saver()
        saver.save(sess, 'my-model')
# =============================================================================
