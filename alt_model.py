import time
import os
import plot_utility
import numpy as np
import string_processing
from lstm import *
import random

SEQUENCE_LEN = 50
LEARN_BATCH_SIZE = 128
GET_INPUT_BATCH_SIZE = 1
GET_INPUT_LEN = 400
TRAIN_BATCHES=100000000

HIDDEN_LEN_1 = 200
HIDDEN_LEN_2 = 400
IN_LEN = string_processing.CHARS_LEN + HIDDEN_LEN_1
OUT_LEN = string_processing.CHARS_LEN

layer1 = LSTM_Layer("model2layer1_200",IN_LEN,HIDDEN_LEN_1)
layer1tan = TanhLayer("tanh_layer34_1",HIDDEN_LEN_1,OUT_LEN)
full_layer1 = TwoLayerLSTM(layer1,layer1tan)

layer2 = LSTM_Layer("model2layer2_400",HIDDEN_LEN_1,HIDDEN_LEN_2)
layer2tan = TanhLayer("tanh_layer34_2",HIDDEN_LEN_2,HIDDEN_LEN_1)
full_layer2 = TwoLayerLSTM(layer2,layer2tan)

optimizer = RMSpropOpt(0.03)

full_layer_learner1 = Learner(full_layer1,optimizer,calc_error_squared,LEARN_BATCH_SIZE,SEQUENCE_LEN)
full_layer_learner2 = Learner(full_layer2,optimizer,calc_error_squared,LEARN_BATCH_SIZE,SEQUENCE_LEN)

def calc_output_fn(raw_inputs,cell1,out1,cell2,out2,full_out):
    in_cells_1 = [[cell1,out1],[]]
    in_cells_2 = [[cell2,out2],[]]
    inputs1 = T.concatenate((raw_inputs,full_out))
    _,[next_cell1,next_outs1] = full_layer1.calc_output(inputs1,in_cells_1)
    inputs2 = next_cell1-cell1
    next_full_outs,[next_cell2,next_outs2] = full_layer2.calc_output(inputs2,in_cells_2)
    return inputs1,inputs2,next_cell1,next_outs1,next_cell2,next_outs2,next_full_outs

def get_batched_zeros(zlen,is_batched):
    if is_batched:
        return (zlen,GET_INPUT_BATCH_SIZE)
    else:
        return (zlen)

def scan_outputs_fn(is_batched):
    text_inputs = T.tensor3("text_inputs_batch") if is_batched else T.matrix("text_inputs")

    [act_inputs1,act_inputs2,_,_,_,_,_],_ = theano.scan(
        calc_output_fn,
        sequences=[text_inputs],
        outputs_info=[
            None,
            None,
            dict(initial=T.zeros(get_batched_zeros(HIDDEN_LEN_1,is_batched),dtype="float32"),taps=[-1]),
            dict(initial=T.zeros(get_batched_zeros(HIDDEN_LEN_1,is_batched),dtype="float32"),taps=[-1]),
            dict(initial=T.zeros(get_batched_zeros(HIDDEN_LEN_2,is_batched),dtype="float32"),taps=[-1]),
            dict(initial=T.zeros(get_batched_zeros(HIDDEN_LEN_2,is_batched),dtype="float32"),taps=[-1]),
            dict(initial=T.zeros(get_batched_zeros(HIDDEN_LEN_1,is_batched),dtype="float32"),taps=[-1])
        ]
    )
    get_inputs_fn = theano.function(
        inputs=[text_inputs],
        outputs=[act_inputs1,act_inputs2]
    )
    return get_inputs_fn

def generate_text_input():
    train_str = string_processing.get_str("data/huck_fin.txt")
    in_vec_list = string_processing.in_vec(train_str)
    in_stack = np.vstack(in_vec_list)
    return in_stack


text_in = generate_text_input()
get_inputs_batched = scan_outputs_fn(True)
get_inputs_unbatched = scan_outputs_fn(False)

def get_random_inputs():
    numvecs = []
    for _ in range(GET_INPUT_BATCH_SIZE):
        start = random.randrange(0,len(text_in)-GET_INPUT_LEN)
        numvecs.append(text_in[start:start+GET_INPUT_LEN])
    nparr = np.dstack(numvecs)
    ac_inpt1,ac_inpt2 = get_inputs_batched(nparr)
    return ac_inpt1,ac_inpt2,nparr

train_time = 0
def train_on(learner,train_fn,inp,exp):
    global train_time
    inproll = np.rollaxis(inp,-1)
    exproll = np.rollaxis(exp,-1)
    for i in range(GET_INPUT_BATCH_SIZE):
        curin = np.transpose(inproll[i])
        curexp = np.transpose(exproll[i])
        for inp,exp in output_trains(learner,curin,curexp,1):
            start = time.clock()
            train_fn(inp,exp)
            train_time += time.clock()-start


def train_model():
    global train_time
    train_fn1 = full_layer_learner1.get_batched_train_pred()
    train_fn2 = full_layer_learner2.get_batched_train_pred()
    for e in range(TRAIN_BATCHES):
        start = time.clock()
        train_time = 0

        in1,in2,text = get_random_inputs()
        train_on(full_layer_learner1,train_fn1,in1,text)
        train_on(full_layer_learner2,train_fn2,in2,in2)

        full_time = time.clock() - start
        print("train_time:",train_time,flush=True)
        print("full_time:",full_time,flush=True)

def predict_model():
    inp1,_ = get_inputs_unbatched(text_in[:1000])
    pred_fn = full_layer_learner1.get_stateful_predict()
    [outputs] = pred_fn(inp1)
    print(string_processing.out_list_to_str(outputs))

def calc_lay1_cells():
    inp1,_ = get_inputs_unbatched(text_in[10000:20000])
    pred_fn = full_layer_learner1.get_stateful_cells()
    [outputs] = pred_fn(inp1)
    #print(outputs)
    #print(np.max(np.abs(outputs)))

def save_text(filename,outtxt):
    with open(filename,"w") as file:
        file.write(outtxt)

def save_full_prediction():
    inp1,_ = get_inputs_unbatched(text_in)
    pred_fn = full_layer_learner1.get_stateful_predict()
    [outputs] = pred_fn(inp1)
    str1 = string_processing.out_list_to_str(outputs)
    save_text("sampled_outputs/model2_full.txt",str1)

#calc_lay1_cells()
save_full_prediction()
#train_model()

def run():
    text_in = generate_text_input()
    NUM_EPOCS = 500
    train(full_layer_learner,text_in,text_in,NUM_EPOCS)
    [outs] = full_layer_learner.get_stateful_predict()(text_in[:1000])
    #np.set_printoptions(threshold=np.inf)
    print(outs)
    print(string_processing.out_list_to_str(outs))
