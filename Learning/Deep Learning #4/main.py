import pandas as pd  # work with data as tables
import numpy as np  # use matrices
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Deep Learning #4 : Linear Regression.')
    parser.add_argument('data_file', help='The file containing datas')
    parser.add_argument('-l', '--learning-rate',
                        dest="lr", default=0.000001,
                        help='The learning rate of the model')
    parser.add_argument('-n', '--number-of-iteration',
                        dest="n", default=1e4,
                        help='The number of iterations')
    parser.add_argument('-s', '--display-step', dest="steps",
                        default=50,
                        help='The display step (?)')
    parser.add_argument('-v', '--verbose', dest="verbose",
                        default=False, action='store_true',
                        help='verbose for debug logs printing')
    args = parser.parse_args()

    # =================================================================
    # Step 1 : load data
    dataframe = pd.read_csv(args.data_file)

    # removing the features we don't want
    dataframe = dataframe.drop(['index', 'price', 'sq_price'], axis=1)
    dataframe = dataframe[0:10]

    # Step 2 - add labels -> nouvelles colonnes de classes qualitatives
    dataframe.loc[:, ('y1')] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    # -> 1 is good ; 0 is bad
    dataframe.loc[:, ("y2")] = dataframe["y1"] == 0  # y2 is the negation of y1
    dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)

    l = max(map(len, str(dataframe).split('\n')))
    print(' selected datas '.center(l, '='))
    print(dataframe)

    # Step 3 - prepare data for tensorflow (tensors)
    # tensor = generic version of vectors and matrices
    # vector = list of numbers [1D tensor]
    # matrix = list of list of numbers [2D tensor]
    # tensor = [list of]^n [nD tensor]

    # convert features to input tensor
    inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
                        # label based indexer

    # convert labels to input tensors
    inputY = dataframe.loc[:, ['y1', 'y2']]

    # Step 4 - parameters
    learning_rate = float(args.lr)
    n = int(args.n)
    display_step = int(args.steps)
    n_samples = inputY.size

    # Step 5 - create the neural net // computation graph
    # for features input tensors, None means any number of exemples
    # placeholders = gateways for data into the computational graph
    x = tf.placeholder(tf.float32, [None, 2])  # 2 features
        # input -> Any x 2

    # create weights : 2 x 2 float matrix
    w = tf.Variable(tf.zeros([2, 2]))
        # 1 input layer (2 neurons) and 1 output layer (2 neurons)
        # 2 x 2 weights

    # add constant
    b = tf.Variable(tf.zeros([2]))
        # bias -> 2 x 1

    # Multiply weights by inputs [first calculation]
    # weights = how the data flows through the graph
    y_values = tf.add(tf.matmul(x, w), b)
        # (X: Any x 2) * (X: 2 x 2) = Any x 2
        # (X*W: Any x 2) + (B: 2 x 1) -> ?

    # apply softmax (activation function)
    y = tf.nn.softmax(y_values)

    print('here')

    # Feed in a matrix of labels
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Step and perform training
    # create the cost function (mean squared error)
    # reduce sum -> computes the sum of elements across all dimensions
    cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * n_samples)

    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(cost)

    # Initialise variables and section
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Training loop
    for i in range(n):
        sess.run(optimizer, feed_dict={x: inputX, y_: inputY})
        # write logs of training
        if args.verbose and (i) % display_step == 0:
            cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
            print("Training step:", '%04d' % (i),
                  "cost =", "{:.9f}".format(cc))
    print("Optimization finished !")
    training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
    print("Training cost=", training_cost, "w=", sess.run(w),
          "b=", sess.run(b))
