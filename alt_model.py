import time
import os
import plot_utility
import numpy as np
import string_processing
from lstm import *

SEQUENCE_LEN = 50
BATCH_SIZE = 256

HIDDEN_LEN_1 = 200
HIDDEN_LEN_2 = 400
IN_LEN = string_processing.CHARS_LEN + HIDDEN_LEN_1
OUT_LEN = string_processing.CHARS_LEN

layer1 = LSTM_Layer("model2layer1_200",IN_LEN,HIDDEN_LEN_1)
layer1tan = TanhLayer("tanh_layer341",HIDDEN_LEN_1,OUT_LEN)
full_layer1 = TwoLayerLSTM(layer1,layer1tan)

layer2 = LSTM_Layer("model2layer2_400",HIDDEN_LEN_1,HIDDEN_LEN_2)
layer2tan = TanhLayer("tanh_layer342",HIDDEN_LEN_2,HIDDEN_LEN_1)
full_layer2 = TwoLayerLSTM(layer2,layer2tan)

optimizer = RMSpropOpt(0.05)

full_layer_learner = Learner(full_layer,optimizer,calc_error_catagorized,BATCH_SIZE,SEQUENCE_LEN)


def calc_output_fn(raw_inputs,cell1,out1,cell2,out2):
    in_cells_1 = [[cell1,out1],[]]
    in_cells_2 = [[cell2,out2],[]]
    inputs1 = T.concatenate(raw_inputs,out2)
    next_outs1,[next_cell1,_] = full_layer1.calc_output(inputs1,in_cells_1)
    inputs2 = next_cell1-in_cells_1
    next_outs2,[next_cell2,_] = full_layer2.calc_output(inputs2,in_cells_2)
    return inputs1,inputs2,next_cell1,next_outs1,next_cell2,next_outs2

def scan_outputs_fn():
    text_inputs = T.matrix("text_inputs")

    [act_inputs1,act_inputs2,_,_,_,_],_ = theano.scan(
        calc_output_fn,
        sequences=[text_inputs],
        outputs_info=[
            None,
            None,
            dict(initial=T.zeros((HIDDEN_LEN_1)),taps=[-1]),
            dict(initial=T.zeros((HIDDEN_LEN_1)),taps=[-1]),
            dict(initial=T.zeros((HIDDEN_LEN_2)),taps=[-1]),
            dict(initial=T.zeros((HIDDEN_LEN_2)),taps=[-1])
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

def run():
    text_in = generate_text_input()
    NUM_EPOCS = 500
    train(full_layer_learner,text_in,text_in,NUM_EPOCS)
    [outs] = full_layer_learner.get_stateful_predict()(text_in[:1000])
    #np.set_printoptions(threshold=np.inf)
    print(outs)
    print(string_processing.out_list_to_str(outs))

def test():
    text_in = generate_text_input()
    expected = text_in[:200]
    [actual] = full_layer_learner.get_stateful_predict()(expected)
    act_text = string_processing.out_list_to_str(actual)
    my_error_fn = error_fn(calc_error_squared)
    errors = []
    skip =  1
    for i in range(0,198,skip):
        [err] = my_error_fn(expected[i+1:i+skip+1],actual[i:i+skip])
        print(err,act_text[i:i+skip])
