#!/usr/bin/python3
# Advanced Scientific Programming with Python - Exercise 3.2

import numpy as np

# a) Create a null vector of size 10 but the fifth value which is 1
a = np.zeros(10) ; a[4] = 1
print("\na:", a)

# b) Create a vector with values ranging from 10 to 49
b = np.arange(10,50)
print("\nb:",b)

# c) Reverse a vector (first element becomes last)
c = b[::-1]
print("\nc:",c)

# d) Create a 3x3 matrix with values ranging from 0 to 8
d = np.arange(0,9).reshape(3,3)
print("\nd:",d)

# e) Find indices of non-zero elements from [1,2,0,0,4,0]
e = np.where(np.array([1,2,0,0,4,0]) != 0)
print("\ne:",e) 

# f) Create a random vector of size 30 and find the mean value
f = np.mean(np.random.rand(30))
print("\nf:",f)

# g) Create a 2d array with 1 on the border and 0 inside
g = np.zeros(shape=(2,4)) ; g[:,0] = 1 ; g[:,-1] = 1
print("\ng:",g)

# h) Create a 8x8 matrix and fill it with a checkerboard pattern
h = np.zeros(shape=(8,8),dtype=np.int32) ; h[np.ix_(np.arange(0,8,2),np.arange(1,8,2))] = 1 ; h[np.ix_(np.arange(1,8,2),np.arange(0,8,2))] = 1
print("\nh:",h)

# i) Create a checkerboard 8x8 matrix using the tile function
i = np.array([[0,1],[1,0]]) ; i = np.tile(i,(4,4))
print("\ni:",i) 

# j) Given a 1D array, negate all elements which are between 3 and 8, in place
j = np.arange(11)
j[np.arange(3,9)] *= -1
print("\nj:",j)

# k) Create a random vector of size 10 and sort it 
k = np.random.random(10)
k = np.sort(k)
print("\nk:",k)

# l) Consider two random array A anb B, check if they are equal
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.array_equal(A,B)
print("\nl:",equal)

# m) How to calculate the square of every number in an array in place (without creating temporaries)?
m = np.arange(10, dtype=np.int32)
print(m.dtype)
m = m**2
print(m.dtype)
print("\nm:",m)

# n) How to get the diagonal of a dot product?
A = np.arange(9).reshape(3,3)
B = A + 1
C = np.dot(A,B)
D = np.diag(C)
print("\nn:",C,D)