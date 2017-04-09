import string
import theano
import numpy as np
import theano.tensor as T
import plot_utility
import itertools
import gc
from WeightBias import WeightBias
from shared_save import RememberSharedVals

#theano.config.optimizer="fast_compile"
#theano.config.scan.allow_gc=True

SEQUENCE_LEN = 6

BATCH_SIZE = 32
EPOCS = 2

GOOD_CHARS = string.ascii_lowercase+" ,.;'-\"\n"
IN_LEN = len(GOOD_CHARS)
OUT_LEN = 120
CELL_STATE_LEN = OUT_LEN
HIDDEN_LEN = OUT_LEN + IN_LEN

FULL_OUT_LEN = len(GOOD_CHARS)

TRAIN_UPDATE_CONST = np.float32(2.0)

inputs = T.tensor3("inputs",dtype="float32")
expected_vec = T.matrix('expected',dtype="float32")

cell_forget_fn = WeightBias("cell_forget", HIDDEN_LEN, CELL_STATE_LEN)
add_barrier_fn = WeightBias("add_barrier", HIDDEN_LEN, CELL_STATE_LEN)
add_cell_fn = WeightBias("add_cell", HIDDEN_LEN, CELL_STATE_LEN)
to_new_output_fn = WeightBias("to_new_hidden", HIDDEN_LEN, CELL_STATE_LEN)

full_output_fn = WeightBias("full_output", CELL_STATE_LEN, FULL_OUT_LEN)

train_plot_util = plot_utility.PlotHolder("train_test")
train_plot_util.add_plot("cell_forget_bias",cell_forget_fn.b)
predict_plot_util = plot_utility.PlotHolder("predict_view")

shared_value_saver = RememberSharedVals('lstm_wbs')

def calc_outputs(in_vec,cell_state,out_vec):
    hidden_input = T.concatenate([out_vec,in_vec],axis=0)

    #first stage, forget some info
    cell_mul_val = T.nnet.sigmoid(cell_forget_fn.calc_output_batched(hidden_input))
    forgot_cell_state = cell_state * cell_mul_val

    #second stage, add some new info
    add_barrier = T.nnet.sigmoid(add_barrier_fn.calc_output_batched(hidden_input))
    this_add_val = T.tanh(add_cell_fn.calc_output_batched(hidden_input))
    added_cell_state = forgot_cell_state + this_add_val * add_barrier

    #third stage, get output
    out_all = T.nnet.sigmoid(to_new_output_fn.calc_output_batched(hidden_input))
    new_output = out_all * T.tanh(added_cell_state)

    return added_cell_state,new_output

def calc_error(expected, actual):
    sqrtdiff = (expected - actual)
    diff = sqrtdiff * sqrtdiff
    error = diff.sum()
    return error

def my_scan(invecs):
    [all_cell,all_out],updates = theano.scan(
        calc_outputs,
        sequences=[invecs],
        outputs_info=[
            dict(initial=T.zeros((OUT_LEN,BATCH_SIZE)),taps=[-1]),
            dict(initial=T.zeros((OUT_LEN,BATCH_SIZE)),taps=[-1])
            ],
        n_steps=SEQUENCE_LEN
    )
    return all_cell[-1],all_out[-1]

out_cell_state,new_out = my_scan(inputs)
predict_plot_util.add_plot("cell_state",out_cell_state)

'''
[out_cell_state,new_out],update = theano.scan(calc_outputs,
                            sequences=[inputs],
                            outputs_info=[
                                dict(initial=T.zeros(CELL_STATE_LEN),taps=[-1]),
                                dict(initial=T.zeros(OUT_LEN),taps=[-1])],
                            truncate_gradient=no_trucation_constant,
                            n_steps=SEQUENCE_LEN)
'''
true_out = T.tanh(full_output_fn.calc_output_batched(new_out))

error = calc_error(true_out,expected_vec)
train_plot_util.add_plot("error_mag",error)

weight_biases = (
    cell_forget_fn.wb_list() +
    add_barrier_fn.wb_list() +
    add_cell_fn.wb_list() +
    to_new_output_fn.wb_list() +
    full_output_fn.wb_list()
)
shared_value_saver.add_shared_vals(weight_biases)
all_grads = T.grad(error,wrt=weight_biases)

def rms_prop_updates():
    DECAY_RATE = np.float32(0.9)
    LEARNING_RATE = np.float32(0.07)
    STABILIZNG_VAL = np.float32(0.00001)

    gsqr = sum(T.sum(g*g) for g in all_grads)

    grad_sqrd_mag = theano.shared(np.float32(0.1),"grad_sqrd_mag")
    shared_value_saver.add_shared_val(grad_sqrd_mag)

    grad_sqrd_mag_update = DECAY_RATE * grad_sqrd_mag + (np.float32(1)-DECAY_RATE)*gsqr

    wb_update_mag = LEARNING_RATE / T.sqrt(grad_sqrd_mag_update + STABILIZNG_VAL)
    train_plot_util.add_plot("update_mag",wb_update_mag)

    wb_update = [(wb,wb - wb_update_mag * grad) for wb,grad in zip(weight_biases,all_grads)]
    return wb_update + [(grad_sqrd_mag,grad_sqrd_mag_update)]

updates = rms_prop_updates()


# Normal SGD
#grads = T.grad(error,wrt=weight_biases)
#updates = [(sh_var,sh_var - TRAIN_UPDATE_CONST*grad) for sh_var,grad in zip(weight_biases,grads)]


predict_fn = theano.function(
    [inputs],
    predict_plot_util.append_plot_outputs([true_out])
)

train_fn = theano.function(
    [inputs,expected_vec],
    train_plot_util.append_plot_outputs([]),
    updates=updates
)

def nice_string(raw_str):
    s = (raw_str.replace("\n\n","\0")
                .replace("\n"," ")
                .replace("\0","\n")
                .replace("”",'"')
                .replace("“",'"')
                .replace("’","'"))
    return "".join(c.lower() for c in s if c.lower() in GOOD_CHARS)
def char_to_vec(c):
    pos = GOOD_CHARS.index(c)
    vec = np.zeros(IN_LEN,dtype="float32")
    vec[pos] = 1.0
    return vec
def in_vec(s):
    return [char_to_vec(c) for c in s]
def get_char(vec):
    ls = list(vec)
    idx = ls.index(max(ls))
    #print(max(ls))
    #print((ls))
    return GOOD_CHARS[idx]
def get_str(filename):
    with open(filename,encoding="utf8") as file:
        return file.read()

train_str = nice_string(get_str("data/wiki_text.txt"))
print(train_str)
instr = np.transpose(np.vstack(in_vec(train_str)))
print(instr.shape)
def output_trains(num_trains):
    for i in range(num_trains):
        for mid in range(SEQUENCE_LEN,len(train_str)-BATCH_SIZE,BATCH_SIZE):
            start = mid - SEQUENCE_LEN
            end = mid + BATCH_SIZE
            in3dtens = np.dstack([instr[:,mid+j:end+j] for j in range(-SEQUENCE_LEN+1,1)] )
            in3dtens = np.rollaxis(in3dtens,-1)
            expected = instr[:,mid:end]
            yield in3dtens,expected
        print("train_epoc"+str(i),flush=True)
        #for i in range(5): gc.collect()

def train():
    for inpt,expect in output_trains(EPOCS):
        output = to_numpys(train_fn(inpt,expect))
        train_plot_util.update_plots(output)
        shared_value_saver.vals_updated()

    shared_value_saver.force_update()

def to_numpys(outlist):
    return [np.asarray(o) for o in outlist]
def predict():
    pred_text = []
    for inp,_ in output_trains(1):
        outputs = to_numpys(predict_fn(inp))
        predict_plot_util.update_plots(outputs)

        pred_text.append(outputs[0])

    text = [let for t in pred_text for let in np.transpose(t) ]

    outtxt = "".join(get_char(v) for v in text)
    return outtxt

train()
print(predict())
