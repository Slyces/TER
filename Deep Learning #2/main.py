import numpy as np
import matplotlib.pyplot as plt

def error(b, m, points):
    N = points.shape[0]
    err = 0
    for (x, y) in points:
        err += (y - m*x - b)**2
    return err / float(points.shape[0])

def gradient_descent(points, b, m, learning_rate, n):
    plot,  = plt.plot(*plotLine(m, b), color='black')
    for i in range(n):
        plot.remove()
        # update b & m
        [b, m] = gradient_step(b, m, points, learning_rate)
        plot, = plt.plot(*plotLine(m, b), color='black')
        plt.pause(0.05)
    return [b, m]

def gradient_step(b: float, m: float, points, learning_rate: float):
    """
    Le gradient représente la dérivée d'une fonction de plusieurs
    variables ; elle pointe vers le bas, c'est à dire vers la minimisation
    de f(*x), où f(*x) est notre erreur
    """
    b_gradient = 0
    m_gradient = 0
    n = points.shape[0]

    for (x, y) in points:
        # on calcule la dérivée partielle de notre fonction
        b_gradient += -(2/n) * (y - m*x - b)
        m_gradient += -(2/n) * x * (y - m*x - b)

    #update de b & m en utilisant les dérivées partielles
    new_b = b + (learning_rate * b_gradient)
    new_m = m + (learning_rate * m_gradient)
    return [new_b, new_m]

def run(file: str, m: float= 0, b: float= 0, lr: float= 1e-4, n: int= 1e4, rand: bool=False):
    """
    Paramètres à définir :
        droite sous la forme y = mx + b
    m  - valeur initiale de m
    b  - valeur initiale de b
    lr - learning rate, ou vitesse de convergeance
    n  - nombre d'itérations
    """
    # Etape 1 - collecter les données
    points = np.genfromtxt(file, delimiter=',', skip_header=1)
    if rand:
        xx = np.array([-0.51, 51.2])
        yy = np.array([0.33, 51.6])
        means = [xx.mean(), yy.mean()]
        stds = [xx.std() / 3, yy.std() / 3]
        corr = 0.8  # correlation
        covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
                [stds[0] * stds[1] * corr, stds[1] ** 2]]

        points = np.random.multivariate_normal(means, covs, 1000).T
        points = points.transpose()

    # Etape 0 - configurer le plot
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Machine learning linear regression")

    x = points[:, 0]
    y = points[:, 1]

    plt.axis((min(x),max(x),min(y),max(y)))
    plt.ion()

    plt.scatter(x, y, s=0.5)

    global plotLine
    def plotLine(m, b):
        return [min(x), max(x)], [m * min(x) + b, m * max(x) + b]

    # Etape 2 - définir les paramètres
    learning_rate = lr
    initial_m = m
    initial_b = b

    # Etape 3 - entrainer le modèle
    print('Starting gradient descent at b = {}, m = {}, error = {}'.format(
        initial_b,
        initial_m,
        error(initial_b, initial_m, points)
    ))
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, n)
    print('Ending point at b = {}, m = {}, error = {}'.format(
        b, m, error(b, m, points)
    ))

    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deep Learning #2 : Linear Regression.')
    parser.add_argument('data_file', help='The file containing datas')
    parser.add_argument('-m', '--initial-m', dest="m", default=1, help='The initial m for y = mx + b')
    parser.add_argument('-b', '--initial-b', dest="b", default=0, help='The initial b for y = mx + b')
    parser.add_argument('-l', '--learning-rate', dest="lr", default=1e-4, help='The learning rate of the model')
    parser.add_argument('-n', '--number-of-iteration', dest="n", default=1e4, help='The number of iterations')
    parser.add_argument('-r', '--random', dest="r", default=False, action='store_true',
                                                                        help='The number of iterations')
    args = parser.parse_args()

    # Running the script
    run(args.data_file, m= float(args.m), b= float(args.b), lr= float(args.lr), n= int(args.n), rand= bool(args.r))
