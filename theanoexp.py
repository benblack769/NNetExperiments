import string
import theano
import numpy as np
import theano.tensor as T
import plot_utility
from WeightBias import WeightBias

#theano.config.optimizer="fast_compile"

inlen = 26
outlen = 26
hiddenlen = 100
train_update_const = 0.3

inputvec = T.dvector('invec')
expectedvec = T.dvector('expected')

hiddenfn = WeightBias("hidden",inlen, hiddenlen)
outfn = WeightBias("out",hiddenlen, outlen)

def logistic(x):
    return 1.0 / (1.0+T.exp(-x))

hidvec = logistic(hiddenfn.calc_output(inputvec))
outvec = logistic(outfn.calc_output(hidvec))

actual = outvec
diff = (expectedvec - actual)**2
error = diff.sum()

outdiff = outfn.update(error,train_update_const)
hiddiff = hiddenfn.update(error,train_update_const)

plotutil = plot_utility.PlotHolder("basic_test")
hidbias_plot = plotutil.add_plot("hidbias",hiddenfn.b)
outbias_plot = plotutil.add_plot("outbias",outfn.b)

predict = theano.function(
        inputs=[inputvec],
        outputs=[outvec]
    )

train = theano.function(
        inputs=[inputvec,expectedvec],
        outputs=plotutil.append_plot_outputs([]),
        updates=hiddiff+outdiff,
    )



def nice_string(s):
    return "".join(c.lower() for c in s if c.lower() in string.ascii_lowercase)
def char_to_vec(c):
    pos = string.ascii_lowercase.index(c)
    vec = np.zeros(26)
    vec[pos] = 1.0
    return vec
def shift_one(c):
    idx = string.ascii_lowercase.index(c)
    nexidx =(idx+1)%26
    return string.ascii_lowercase[nexidx]
def in_vec(s):
    return [char_to_vec(c) for c in s]
def expect_vec(s):
    return [char_to_vec(shift_one(c)) for c in s]
def get_char(vec):
    ls = list(vec)
    idx = ls.index(max(ls))
    #print(max(ls))
    #print((ls))
    return string.ascii_lowercase[idx]
def get_str(filename):
    with open(filename) as file:
        return file.read()

train_str = nice_string(get_str("data/test_text.txt"))
instr = in_vec(train_str)
exp_str = expect_vec(train_str)
#print(train_str)
for _ in range(30):
    for inp, ex in zip(instr,exp_str):
        pass
        output = train(inp,ex)
        plotutil.update_plots(output)

predtext = [predict(c) for c in instr]
outtxt = "".join(get_char(v[0]) for v in predtext)
#print(predtext)
print(outtxt)
#print(outfn.W.get_value())
