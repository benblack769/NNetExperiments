import scipy.linalg.blas
import theano
import theano.tensor as T
from theano import function

#theano.config.gcc.cxxflags = "asdsion -std=gnu++11"

x = T.dscalar('x')
y = T.dscalar('y')
z=x+y
f = function([x,y],z)
val = f(1,2)
print(val)
