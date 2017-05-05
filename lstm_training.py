import time
import os
import plot_utility
import numpy as np
import string_processing
from lstm import *

SEQUENCE_LEN = 50
BATCH_SIZE = 256

IN_LEN = string_processing.CHARS_LEN
HIDDEN_LEN_1 = 400
HIDDEN_LEN_2 = 200
OUT_LEN = string_processing.CHARS_LEN

layer1 = LSTM_Layer("layer1",IN_LEN,HIDDEN_LEN_1)
layer2 = LSTM_Layer("layer9",HIDDEN_LEN_1,HIDDEN_LEN_2)
layer3 = LSTM_Layer("tanh_layer",HIDDEN_LEN_2,OUT_LEN)
full_layer = TwoLayerLSTM(TwoLayerLSTM(layer1,layer2),layer3)
layer_names = [l.name for l in full_layer.get_weight_biases()]

optimizer = RMSpropOpt(0.06)

full_layer_learner = Learner(full_layer,optimizer,calc_error_squared,BATCH_SIZE,SEQUENCE_LEN)

def generate_text_input():
    train_str = string_processing.get_str("data/huck_fin.txt")
    in_vec_list = string_processing.in_vec(train_str)
    in_stack = np.vstack(in_vec_list)
    return in_stack

text_in = generate_text_input()
NUM_EPOCS = 100
train(full_layer_learner,text_in,text_in,NUM_EPOCS)
[outs] = full_layer_learner.get_stateful_predict()(text_in[:1000])
#np.set_printoptions(threshold=np.inf)
print(outs)
print(string_processing.out_list_to_str(outs))
