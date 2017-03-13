import numpy as np

def error(float b, float m, points):
    cdef int N
    N = points.shape[0]
    cdef float err, y, x
    err = 0
    for (y, x) in points:
        err += y - m*x - b
    return err / N

def gradient_descent(points, float b, float m, float learning_rate, int n):
    cdef int i
    for i in range(n):
        # update b & m
        [b, m] = gradient_step(b, m, points, learning_rate)
    return [b, m]

def gradient_step(float b, float m, points, float learning_rate):
    """
    Le gradient représente la dérivée d'une fonction de plusieurs
    variables ; elle pointe vers le bas, c'est à dire vers la minimisation
    de f(*x), où f(*x) est notre erreur
    """
    cdef float b_gradient, m_gradient, new_b, new_m, y, x
    cdef int n
    b_gradient = 0
    m_gradient = 0
    n = points.shape[0]

    for (y, x) in points:
        # on calcule la dérivée partielle de notre fonction
        b_gradient += -(2/n) * (y - m*x - b)
        m_gradient +=  (2/n) * x * (y - m*x - b)

    #update de b & m en utilisant les dérivées partielles
    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)
    return [new_b, new_m]


def run(file: str, m: float= 0, b: float= 0, lr: float= 1e-4, n: int= 1e4):
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deep Learning #2 : Linear Regression.')
    parser.add_argument('data_file', help='The file containing datas')
    parser.add_argument('-m', '--initial-m', dest="m", default=0, help='The initial m for y = mx + b')
    parser.add_argument('-b', '--initial-b', dest="b", default=0, help='The initial b for y = mx + b')
    parser.add_argument('-l', '--learning-rate', dest="lr", default=1e-4, help='The learning rate of the model')
    parser.add_argument('-n', '--number-of-iteration', dest="n", default=1e4, help='The number of iterations')
    args = parser.parse_args()

    # Running the script
    run(args.data_file, m= args.m, b= args.b, lr= args.lr, n= int(args.n))
