from lstm import cell_forget_fn,add_barrier_fn,add_cell_fn,to_new_output_fn,full_output_fn,CELL_STATE_LEN,OUT_LEN
import numpy as np
from WeightBias import NP_WeightBias
import string_processing

def np_sigmoid(vec):
    return 1.0/(1.0+np.exp(-vec))

def np_calc_output(inp_vec,cell_state,output_vec,np_cell_forget,np_add_barrier,np_add_cell,np_new_output):
    #print(output_vec,inp_vec)
    hidden_input = np.concatenate([output_vec,inp_vec])

    #first stage, forget some info
    cell_mul_val = np_sigmoid(np_cell_forget.calc_output_batched(hidden_input))
    forgot_cell_state = cell_state * cell_mul_val

    #second stage, add some new info
    add_barrier = np_sigmoid(np_add_barrier.calc_output_batched(hidden_input))
    this_add_val = np.tanh(np_add_cell.calc_output_batched(hidden_input))
    added_cell_state = forgot_cell_state + this_add_val * add_barrier

    #third stage, get output
    out_all = np_sigmoid(np_new_output.calc_output_batched(hidden_input))
    new_output = out_all * np.tanh(added_cell_state)
    return added_cell_state,new_output


def stateful_predict(input_list):
    cell_state = np.zeros(CELL_STATE_LEN)
    output_vec = np.zeros(OUT_LEN)
    np_cell_forget = NP_WeightBias(cell_forget_fn)
    np_add_barrier = NP_WeightBias(add_barrier_fn)
    np_add_cell = NP_WeightBias(add_cell_fn)
    np_new_output = NP_WeightBias(to_new_output_fn)
    np_full_output = NP_WeightBias(full_output_fn)
    all_cell_states = []
    all_outputs = []
    i = 0
    for inp_vec in input_list:
        cell_state,output_vec = np_calc_output(inp_vec,cell_state,output_vec,np_cell_forget,np_add_barrier,np_add_cell,np_new_output)
        all_cell_states.append(cell_state)
        all_outputs.append(np.tanh(np_full_output.calc_output(output_vec)))
        if i%100 == 0:
            print(i,flush=True)
        i += 1
    return all_cell_states,all_outputs


train_str = string_processing.get_str("data/huck_fin.txt")[:1000]
in_vec_list = string_processing.in_vec(train_str)
out_cells,out_outputs = stateful_predict(in_vec_list)

outtxt = string_processing.out_list_to_str(out_outputs)

print(outtxt)
