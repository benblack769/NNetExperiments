import string
import theano
import numpy as np
import theano.tensor as T
import plot_utility
import itertools
from WeightBias import WeightBias

SEQUENCE_LEN = 6

BATCH_SIZE = 16

IN_LEN = 26
OUT_LEN = 26
CELL_STATE_LEN = OUT_LEN
HIDDEN_LEN = OUT_LEN + IN_LEN


TRAIN_UPDATE_CONST = 2

inputs = T.matrix("inputs")
expected_vec = T.vector('expected')

cell_forget_fn = WeightBias("cell_forget", HIDDEN_LEN, CELL_STATE_LEN)
add_barrier_fn = WeightBias("add_barrier", HIDDEN_LEN, CELL_STATE_LEN)
add_cell_fn = WeightBias("add_cell", HIDDEN_LEN, CELL_STATE_LEN)
to_new_output_fn = WeightBias("to_new_hidden", HIDDEN_LEN, CELL_STATE_LEN)


train_plot_util = plot_utility.PlotHolder("train_test")
train_plot_util.add_plot("cell_forget_bias",cell_forget_fn.b)
predict_plot_util = plot_utility.PlotHolder("predict_view")

def sigma(x):
    return 1.0 / (1.0+T.exp(-x))

def calc_outputs(in_vec,cell_state,out_vec):
    hidden_input = T.concatenate([out_vec,in_vec])

    #first stage, forget some info
    cell_mul_val = sigma(cell_forget_fn.calc_output(hidden_input))
    forgot_cell_state = cell_state * cell_mul_val

    #second stage, add some new info
    add_barrier = sigma(add_barrier_fn.calc_output(hidden_input))
    this_add_val = T.tanh(add_cell_fn.calc_output(hidden_input))
    added_cell_state = forgot_cell_state + this_add_val * add_barrier

    #third stage, get output
    out_all = sigma(to_new_output_fn.calc_output(hidden_input))
    new_output = out_all * T.tanh(added_cell_state)

    return added_cell_state,new_output

def calc_error(expected, actual):
    diff = (expected - actual)**2
    error = diff.sum()
    return error

def my_scan(invecs):
    prev_out = T.zeros(OUT_LEN)
    prev_cell = T.zeros(CELL_STATE_LEN)
    for x in range(SEQUENCE_LEN):
        prev_cell,prev_out = calc_outputs(invecs[x],prev_cell,prev_out)
        #prev_out = prev_out.dimshuffle((0,1))#.dimshuffle((1,))
    predict_plot_util.add_plot("cell_state",prev_cell)
    return prev_cell,prev_out

out_cell_state,new_out = my_scan(inputs)

'''
[out_cell_state,new_out],update = theano.scan(calc_outputs,
                            sequences=[inputs],
                            outputs_info=[
                                dict(initial=T.zeros(CELL_STATE_LEN),taps=[-1]),
                                dict(initial=T.zeros(OUT_LEN),taps=[-1])],
                            truncate_gradient=no_trucation_constant,
                            n_steps=SEQUENCE_LEN)
'''
true_out = new_out

error = calc_error(true_out,expected_vec)

weight_biases = (
    cell_forget_fn.wb_list() +
    add_barrier_fn.wb_list() +
    add_cell_fn.wb_list() +
    to_new_output_fn.wb_list()
)
grads = T.grad(error,wrt=weight_biases)
updates = [(sh_var,sh_var - TRAIN_UPDATE_CONST*grad) for sh_var,grad in zip(weight_biases,grads)]

predict_fn = theano.function(
    [inputs],
    predict_plot_util.append_plot_outputs([true_out])
)

train_fn = theano.function(
    [inputs,expected_vec],
    train_plot_util.append_plot_outputs([]),
    updates=updates
)



def nice_string(s):
    return "".join(c.lower() for c in s if c.lower() in string.ascii_lowercase)
def char_to_vec(c):
    pos = string.ascii_lowercase.index(c)
    vec = np.zeros(26)
    vec[pos] = 1.0
    return vec
def in_vec(s):
    return [char_to_vec(c) for c in s]
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
print(train_str)
instr = in_vec(train_str)
for i in range(40):
    for end in range(SEQUENCE_LEN,len(instr)-1):
        start = end - SEQUENCE_LEN
        inmat = np.vstack(instr[start:end])
        expected = instr[end+1]
        output = train_fn(inmat,expected)
        train_plot_util.update_plots(output)
    print("train_epoc"+str(i),flush=True)

pred_text = []
for end in range(SEQUENCE_LEN,len(instr)-1):
    start = end - SEQUENCE_LEN
    inmat = np.vstack(instr[start:end])
    outputs = predict_fn(inmat)
    predict_plot_util.update_plots(outputs)
    pred_text.append(outputs[0])


outtxt = "".join(get_char(v) for v in pred_text)
#print(predtext)
print(outtxt)
#print(outfn.W.get_value())
