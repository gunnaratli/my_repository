import numpy as np
import cupy

for N in [128, 256, 512, 1024, 2048]:

	# numpy array of integers
 	arr_n = np.random.uniform(0, 100, size=(N, N)).astype(np.float32)

 	# numpy array of float32
 	arr_n = np.random.uniform(0, 100, size=(N, N)).astype(np.float32)

 	# Convert the numpy array to cude array
 	arr_c = cupy.array(arr_n)

	# Compute a 2D Fourier transformation and compare the times 	
  	print("Times achieved for array size %4d" %N)
  	%timeit np.fft.fft2(arr_n)
  	%timeit cupy.fft.fft2(arr_c)



"""
Thoughts: cupy outperforms already from the start. For numpy, the operation takes slightly longer using float32 than it does using int64


From Google colab:

With numpy.int64
Times achieved for array size  128
226 µs ± 4.63 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
175 µs ± 59.4 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
Times achieved for array size  256
1.13 ms ± 257 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
74.9 µs ± 10.8 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
Times achieved for array size  512
5.07 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
252 µs ± 2.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
Times achieved for array size 1024
26.5 ms ± 581 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
1.09 ms ± 4.65 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
Times achieved for array size 2048
158 ms ± 21 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
4.2 ms ± 9.45 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


With numpy.float32:
Times achieved for array size  128
225 µs ± 5.34 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
115 µs ± 46.4 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
Times achieved for array size  256
1.65 ms ± 315 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
108 µs ± 17.7 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
Times achieved for array size  512
7.34 ms ± 1.17 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
70.2 µs ± 2.33 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
Times achieved for array size 1024
32.7 ms ± 6.85 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
229 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
Times achieved for array size 2048
165 ms ± 29 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
958 µs ± 1.36 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
"""