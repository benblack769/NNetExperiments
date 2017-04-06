import string
import theano
import numpy as np
import theano.tensor as T
import plot_utility
from WeightBias import WeightBias

inlen = 26
outlen = 26
hiddenlen = 100
batch_size = 32
train_update_const = 0.3/batch_size

inputvec = T.dmatrix('invec')
expectedvec = T.dmatrix('expected')

hiddenfn = WeightBias("hidden",inlen, hiddenlen)
outfn = WeightBias("out",hiddenlen, outlen)

def logistic(x):
    return 1.0 / (1.0+T.exp(-x))

hidvec = logistic(hiddenfn.calc_output_batched(inputvec))
outvec = logistic(outfn.calc_output_batched(hidvec))

actual = outvec
diff = (expectedvec - actual)**2
error = diff.sum()

all_wb = outfn.wb_list() + hiddenfn.wb_list()
all_grad = T.grad(error,wrt=all_wb)

def rms_prop_updates():
    DECAY_RATE = 0.9
    LEARNING_RATE = 0.1
    STABILIZNG_VAL = 0.00001

    gsqr = sum(T.sum(g*g) for g in all_grad)

    grad_sqrd_mag = theano.shared(0.1,"grad_sqrd_mag")

    grad_sqrd_mag_update = DECAY_RATE * grad_sqrd_mag + (1-DECAY_RATE)*gsqr

    wb_update_mag = LEARNING_RATE / T.sqrt(grad_sqrd_mag_update + STABILIZNG_VAL)
    plotutil.add_plot("update_mag",grad_sqrd_mag_update)

    wb_update = [(wb,wb - wb_update_mag * grad) for wb,grad in zip(all_wb,all_grad)]
    return wb_update + [(grad_sqrd_mag,grad_sqrd_mag_update)]

plotutil = plot_utility.PlotHolder("rmsprop_test")
plotutil.add_plot("hidbias",hiddenfn.b)
plotutil.add_plot("outbias",outfn.b)

updates = rms_prop_updates()

predict = theano.function(
        inputs=[inputvec],
        outputs=[outvec]
    )

train = theano.function(
        inputs=[inputvec,expectedvec],
        outputs=plotutil.append_plot_outputs([]),
        updates=updates
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
def to_single_mat(vecs):
    return np.transpose(np.vstack(vecs))
def matrixfy(vecs,matlen):
    return [to_single_mat(vecs[i:i+matlen]) for i in range(0,len(vecs),matlen)]
def break_mat_list(matls):
    res = []
    for mat in matls:
        transmat = np.transpose(mat)
        for vec in transmat:
            res.append(vec)
    return res
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
instr = matrixfy(in_vec(train_str),batch_size)
exp_str = matrixfy(expect_vec(train_str),batch_size)
#print(instr)
for _ in range(300):
    for inp, ex in zip(instr,exp_str):
        pass
        output = train(inp,ex)
        plotutil.update_plots(output)

predtext = [predict(c)[0] for c in instr]
pred_vecs = break_mat_list(predtext)
#print(pred_vecs)
#print(predtext)
outtxt = "".join(get_char(v) for v in pred_vecs)
#print(predtext)
print(outtxt)
#print(outfn.W.get_value())
