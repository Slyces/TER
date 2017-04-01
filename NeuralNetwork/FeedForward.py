import tensorflow as tf, numpy as np, pandas
from NeuralNetwork import Model


class FeedForward(Model.NeuralNetwork):
    def create_model(self, X, w_h: list, w_b: list = None):
        """
        w_h et w_b deux listes de même taille, minimum 2.
        w_b peut être optionnel.
        Le dernier élément est le noeud de sortie, les autres seront des layers
        """
        assert len(w_h) >= 2
        for i in range(len(w_h) - 1):
            print(i)
            last_layer = self.vars['h%s' % (i - 1)] if i > 0 else X
            if w_b:
                self.vars['h%s' % i] = tf.add(
                    tf.matmul(last_layer, w_h[i]), w_b[i])
            else:
                self.vars['h%s' % i] = tf.matmul(last_layer, w_h[i])
        if w_b:
            self.vars['model'] = tf.add(
                tf.matmul(self.vars['h%s' % (len(w_h) - 2)], w_h[-1]), w_b[-1])
        else:
            self.vars['model'] = tf.matmul(self.vars['h%s' % (len(w_h) - 2)], w_h[-1])

    def build_model(self):

        self.vars['X'] = tf.placeholder("float", [None, 40], name="X")
        self.vars['Y'] = tf.placeholder("float", [None, 4], name="Y")

        w_h = [
            self.init_weights([40, 20]),
            self.init_weights([20, 4])
        ]

        w_b = [
            tf.Variable(tf.zeros([20])),
            tf.Variable(tf.zeros([4]))
        ]

        self.create_model(self.vars['X'], w_h, w_b)

        self.vars['cost'] = tf.reduce_sum(tf.square(self.vars['model'] - self.vars['Y']))
        self.vars['train'] = tf.train.GradientDescentOptimizer(0.001).minimize(self.vars['cost'])

    def performance(self):
        return str(self.sess.run(self.vars['cost'], feed_dict={self.vars['X']: self.data['teX'],
                                                               self.vars['Y']: self.data['teY']}))

    def train_batch(self, start, end):
        self.sess.run(self.vars['train'], feed_dict={self.vars['X']: self.data['TrX'][start:end],
                                                     self.vars['Y']: self.data['TrY'][start:end]})

    def gather_data(self):
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
        return Xdatas, Ydatas

if __name__ == '__main__':
    feed = FeedForward('memorry/my-model')