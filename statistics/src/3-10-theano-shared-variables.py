import theano.tensor as T
from theano import function, shared

state = shared(0)
x = T.iscalar("x")
acc = function([x], state**2, updates=[(state, state+x)])

print(acc(0) + acc(1) + acc(10) + acc(11))