#!/usr/bin/python3

# Program to multiply two matrices using nested loops
import numpy as np
import time

def matmult(N):

    # Generate random matrices with the given dimensions
    X = np.random.randint(0, 101, size=(N, N))
    Y = np.random.randint(0, 101, size=(N, N+1))

    return np.dot(X, Y)  



# Multiply two matrices using nested numpy
def main():
    N = 250
    t1 = time.time()
    result = matmult(N)
    t2 = time.time()
    print("Elapsed time: ", t2-t1)
    print(result.shape)


if __name__ == "__main__":
    main()