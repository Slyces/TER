import time
def benchmark(f):
    def decorate(*args, **kwargs):
        t0 = time.time()
        _ = f(*args, **kwargs)
        t1 = time.time()
        print("Benchmarked {} s".format(t1 - t0))
        return _
    return decorate

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

    import main as main
    import cmain as cmain
    print("Main :")
    benchmark(main.run)(args.data_file, m= args.m, b= args.b, lr= args.lr, n= int(args.n))
    print("cMain :")
    benchmark(cmain.run)(args.data_file, m=args.m, b=args.b, lr=args.lr, n=int(args.n))