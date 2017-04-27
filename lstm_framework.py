import lstm
import time
import os
import plot_utility
import numpy as np
import string_processing

SEQUENCE_LEN = 50

BATCH_SIZE = 128

def gen_lstm_with(lstm_name,in_stack,exp_stack,inter_len):
    IN_LEN = in_stack.shape[1]
    OUT_LEN = inter_len
    CELL_STATE_LEN = OUT_LEN
    HIDDEN_LEN = OUT_LEN + IN_LEN

    FULL_OUT_LEN = exp_stack.shape[1]

    my_lstm = lstm.LSTM(lstm_name,SEQUENCE_LEN,IN_LEN,OUT_LEN,FULL_OUT_LEN,BATCH_SIZE)

    return my_lstm

def stateful_predict():
    [outs] = my_lstm.state_predict(in_stack)
    return string_processing.out_list_to_str(outs)

base_input_filename = "saved_cells/huck_fin_input_data.npy"
first_stage_cell_filename = "saved_cells/huck_fin_small_cell.npy"
second_stage_output_filename = "saved_cells/huck_fin_stage1_output.npy"


#in_base_stack = np.load(base_input_filename)
#in_first_stack = np.tanh(np.load(first_stage_cell_filename))
#in_stack2 = np.concatenate((in_base_stack,in_first_stack),axis=1)
#lstm_stage2 = gen_lstm_with("huck_fin_stage2",in_stack2,in_first_stack,400)
#lstm_stage2.train(in_stack2,in_first_stack,100)
