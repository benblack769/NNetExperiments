import string
import scipy.linalg.blas
import theano
import numpy as np
import theano.tensor as T
import plot_utility

theano.config.optimizer="fast_compile"
class WeightBias:
    def __init__(self,name,in_len,out_len):
        self.name = name

        rand_weight_vals = np.random.randn(in_len*out_len)/in_len**(0.5**0.5)
        rand_weight = np.reshape(rand_weight_vals,(out_len,in_len))
        self.W = theano.shared(rand_weight,name=self.weight_name())

        bias_init = np.zeros(out_len)
        self.b = theano.shared(bias_init,name=self.bias_name())

    def calc_output(self,in_vec):
        return self.b + T.dot(self.W,in_vec)

    def bias_name(self):
        return self.name+"b"
    def weight_name(self):
        return self.name+"W"
    def wb_list(self):
        return [self.W,self.b]
    def update(self,error):
        Wg,bg = T.grad(error,self.wb_list())

        c = train_update_const
        return [(self.W,self.W-Wg*c),(self.b,self.b-bg*c)]

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

outdiff = outfn.update(error)
hiddiff = hiddenfn.update(error)

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

train_str = nice_string(get_str("test_text.txt"))
instr = in_vec(train_str)
exp_str = expect_vec(train_str)
#print(train_str)
for _ in range(10):
    for inp, ex in zip(instr,exp_str):
        pass
        output = train(inp,ex)
        plotutil.update_plots(output)

predtext = [predict(c) for c in instr]
outtxt = "".join(get_char(v[0]) for v in predtext)
#print(predtext)
print(outtxt)
#print(outfn.W.get_value())
