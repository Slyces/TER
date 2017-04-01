"""
https://www.oreilly.com/learning/hello-tensorflow
"""
import tensorflow as tf

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Deep Learning #4 : Linear Regression.')
    # parser.add_argument('data_file', help='The file containing datas')
    # parser.add_argument('-l', '--learning-rate',
    #                     dest="lr", default=0.000001,
    #                     help='The learning rate of the model')
    # parser.add_argument('-n', '--number-of-iteration',
    #                     dest="n", default=1e4,
    #                     help='The number of iterations')
    # parser.add_argument('-s', '--display-step', dest="steps",
    #                     default=50,
    #                     help='The display step (?)')
    parser.add_argument('-v', '--verbose', dest="verbose",
                        default=False, action='store_true',
                        help='verbose for debug logs printing')
    args = parser.parse_args()

    # Code condensé !!!
    import tensorflow as tf

    x = tf.constant(1.0, name='input')
    w = tf.Variable(0.8, name='weight')
    y = tf.multiply(w, x, name='output')
    y_ = tf.constant(0.0, name='correct_value')
    loss = tf.pow(y - y_, 2, name='loss')
    train_step = tf.train.GradientDescentOptimizer(0.025
                                                   ).minimize(loss)

    for value in [x, w, y, y_, loss]:
        tf.summary.scalar(value.op.name, value)

    summaries = tf.summary.merge_all()

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter('log_simple_stats',
                                           sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(100):
        summary_writer.add_summary(sess.run(summaries), i)
        sess.run(train_step)