#!/usr/bin/python3

import simple_math

def test_simple_add():
	assert simple_math.simple_add(1,1) == 2
	assert simple_math.simple_add(10,17) == 27

def test_simple_sub():
	assert simple_math.simple_sub(3,2) == 1

def test_simple_mult():
    assert simple_math.simple_mult(3,4) == 12

def test_simple_div():
	assert simple_math.simple_div(12,6) == 2

def test_poly_first():
	assert simple_math.poly_first(1,1,1) == 2

def test_poly_second():
	assert simple_math.poly_second(1,1,1,1) == 3