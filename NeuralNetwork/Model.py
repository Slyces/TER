import tensorflow as tf, numpy as np


class NeuralNetwork(object):
    """ Neural Network class to support tensorflow build models """

    @staticmethod
    def init_weights(shape, name="weights"):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def __init__(self, save_path: str = ""):
        self.path = save_path
        self.sess = tf.Session()
        self.vars = {}  # dict to store inner variables of the model
        self.data = {}
        self.build_model()
        print(self.vars)
        self.saver = tf.train.Saver(self.vars)
        self.split_data(*self.gather_data())

    def create_model(self, *args):
        """
        Creates the model structure

        Inner variables must be placed in self.vars [dict]

        This method must implement :
            - self.vars['X']
            - self.vars['Y']
            - self.vars['train']
            - self.vars['cost']
            - self.vars['model'] # Last layer
        """
        # Save inner variables to self.vars
        raise NotImplementedError

    def build_model(self):
        """ Method calling create_model with the right parameters """
        pass

    def save_model(self):
        """ Saves the model using tensorflow's saving protocol """
        return self.saver.save(self.sess, self.path)

    def load_model(self):
        return self.saver.restore(self.sess, self.path)

    def train_batch(self, start: int, end: int):
        """ Implements the training of one batch """
        raise NotImplementedError

    def train(self, n: int, train_len, batches: int = 1,
              save_freq: int = 0, perf_freq: int= 0,new=True):
        """
        Method for training the network

        put save frequency to 0 to save only at the end
        """
        if new:
            tf.global_variables_initializer().run(session=self.sess)
        else:
            self.load_model()
        for i in range(n):
            for start, end in zip(range(0, train_len, batches), range(batches, train_len + 1, batches)):
                self.train_batch(start, end)
            if save_freq and i % save_freq == 0:
                self.save_model()
            if perf_freq and i % perf_freq == 0:
                print(i, self.performance())
        self.save_model()

    def performance(self):
        """
        A method to return a string of the performance of your model
        (cost, % of predictions ...)
        """
        raise NotImplementedError

    def split_data(self, Xdatas, Ydatas):
        """ Splits the data between train and test, returning trX, trY, teX, teY"""
        p_train = 0.75
        indexes = np.random.rand(len(Xdatas)) < p_train
        self.data['trX'] = Xdatas[indexes]
        self.data['trY'] = Ydatas[indexes]
        self.data['teX'] = Xdatas[~indexes]
        self.data['teY'] = Ydatas[~indexes]

    def gather_data(self):
        """ returns the X datas and Y datas, which are then splitted and stockes in self.data """
        raise NotImplementedError
