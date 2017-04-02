import tensorflow as tf
import numpy as np
import pandas
import sqlite3
from hello_world.Scripts.NeuralNetwork import Model
from os.path import join
from django_world.settings import DATABASE_PATH


class FeedForward(Model.NeuralNetwork):
    def create_model(self, X, w_h: list, w_b: list = None):
        """
        w_h et w_b deux listes de même taille, minimum 2.
        w_b peut être optionnel.
        Le dernier élément est le noeud de sortie, les autres seront des layers
        """
        assert len(w_h) >= 2
        for i in range(len(w_h) - 1):
            last_layer = self.ops['h%s' % (i - 1)] if i > 0 else X
            if w_b:
                self.ops['h%s' % i] = tf.add(
                    tf.matmul(last_layer, w_h[i]), w_b[i])
            else:
                self.ops['h%s' % i] = tf.matmul(last_layer, w_h[i])
        if w_b:
            self.ops['model'] = tf.add(
                tf.matmul(self.ops['h%s' % (len(w_h) - 2)], w_h[-1]), w_b[-1])
        else:
            self.ops['model'] = tf.matmul(self.ops['h%s' % (len(w_h) - 2)], w_h[-1])

    def build_model(self):

        self.placeholders['X'] = tf.placeholder("float", [None, 40], name="X")
        self.placeholders['Y'] = tf.placeholder("float", [None, 4], name="Y")

        self.vars['w_0'] = self.init_weights([40, 40])
        self.vars['w_o'] = self.init_weights([40, 4])

        self.vars['b_0'] = tf.Variable(tf.zeros([40]))
        self.vars['b_o'] = tf.Variable(tf.zeros([4]))

        w_h = [self.vars['w_0'], self.vars['w_o']]
        w_b = [self.vars['b_0'], self.vars['b_o']]

        self.create_model(self.placeholders['X'], w_h, w_b)

        self.ops['cost'] = tf.reduce_sum(tf.square(self.ops['model'] - self.placeholders['Y']))
        self.ops['train'] = tf.train.GradientDescentOptimizer(0.001).minimize(self.ops['cost'])

    def performance(self):
        return str(self.sess.run(self.ops['cost'], feed_dict={self.placeholders['X']: self.data['teX'],
                                                              self.placeholders['Y']: self.data['teY']}))

    def train_batch(self, start, end):
        self.sess.run(self.ops['train'], feed_dict={self.placeholders['X']: self.data['trX'][start:end],
                                                    self.placeholders['Y']: self.data['trY'][start:end]})

    def gather_data(self):
        # processed = pandas.read_csv("processed.csv", sep=",")

        with sqlite3.connect(DATABASE_PATH) as con:
            processed = pandas.read_sql("SELECT * FROM historic_indexes;", con)

        n = len(processed.DateTime.values) // 11

        # On découpe par tranches de 11
        Xdatas = np.array([
                              np.concatenate([processed.loc[i * 11:i * 11 + 9, ('Nasdaq', 'Dowjones',
                                                                                'SnP500', 'Rates')[j]]
                                              for j in range(4)]) for i in range(n)
                              ])
        Ydatas = np.array([[processed.loc[i * 11 + 10, ('Nasdaq', 'Dowjones',
                                                        'SnP500', 'Rates')[j]]
                            for j in range(4)] for i in range(n)
                           ])
        return Xdatas, Ydatas

    def predict(self, label):
        self.load_model()
        with sqlite3.connect(DATABASE_PATH) as con:
            processed = pandas.read_sql("SELECT * FROM historic_indexes;", con)

        X = np.array([
                         np.concatenate([processed.loc[i:i + 9, ('Nasdaq', 'Dowjones',
                                                                 'SnP500', 'Rates')[j]]
                                         for j in range(4)]) for i in range(len(processed.Id) - 11)
                         ])
        Y = np.array([[processed.loc[i + 10, ('Nasdaq', 'Dowjones',
                                              'SnP500', 'Rates')[j]]
                       for j in range(4)] for i in range(len(processed.Id) - 11)
                      ])

        request = ("INSERT INTO predicted_indexes(Label, DateTime, Nasdaq, Dowjones, SnP500, Rates)"
                   "VALUES (?,?,?,?,?,?);")

        with sqlite3.connect(DATABASE_PATH) as con:
            cursor = con.cursor()
            for i in range(len(X)):
                data = [label, processed.loc[i+11, 'DateTime']] + [float(x) for x in
                    self.sess.run(self.ops['model'], feed_dict={
                    self.placeholders['X']: X[i].reshape((1, 40)),
                    self.placeholders['Y']: Y[i].reshape((1, 4))
                })[0]]
                cursor.execute(request, data)
            con.commit()


if __name__ == '__main__':
    feed = FeedForward('memorry/my-model')
    # print(feed.train(100000, 50, perf_freq=100, new=False))
    feed.predict("2")
