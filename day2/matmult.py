#!/usr/bin/python3

# Program to multiply two matrices using nested loops
import random
import time

def matmult(N):

    # NxN matrix
    X = []
    for i in range(N):
        X.append([random.randint(0,100) for r in range(N)])

    # Nx(N+1) matrix
    Y = []
    for i in range(N):
        Y.append([random.randint(0,100) for r in range(N+1)])

    # result is Nx(N+1)
    result = []
    for i in range(N):
        result.append([0] * (N+1))

    # iterate through rows of X
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]

    for r in result:
        print(r)

    return result


# Multiply two matrices using nested loops
def main():

    N = 250
    t1 = time.time()
    result = matmult(N)
    t2 = time.time()
    print("Elapsed time: ", t2-t1)
    print(len(result))


if __name__ == "__main__":
    main()