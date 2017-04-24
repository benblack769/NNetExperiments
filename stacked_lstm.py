import lstm_framework
import numpy as np
import string_processing

def generate_text_input():
    train_str = string_processing.get_str("data/huck_fin.txt")
    in_vec_list = string_processing.in_vec(train_str)
    in_stack = np.vstack(in_vec_list)
    np.save("saved_cells/huck_fin_input_data",in_stack)

def cells_to_outputs(cells):
    #base_constant = 0.6
    #return np.tanh(cells*base_constant)
    sub_cells = cells[1:]-cells[:-1]
    all_cells = np.concatenate((np.zeros((1,cells.shape[1]),dtype="float32"),sub_cells))
    return all_cells

#### Base LSTM training
def run_basic_lstm():
    in_base_stack = np.load(lstm_framework.base_input_filename)
    my_lstm = lstm_framework.gen_lstm_with("wide_huck_fin_small",in_base_stack,in_base_stack,200)
    #[out_vals] = my_lstm.state_predict(in_base_stack[:3000])
    #print(string_processing.out_list_to_str(out_vals))

    #my_lstm.train(in_base_stack,in_base_stack,50)
    #my_lstm.save_stateful_cells(first_stage_cell_filename,in_stack)

def run_level2_lstm():
    in_base_stack = np.load(lstm_framework.base_input_filename)
    in_first_stack = cells_to_outputs(np.load(lstm_framework.first_stage_cell_filename))

    in_stack2 = np.concatenate((in_base_stack,in_first_stack),axis=1)

    lstm_stage2 = lstm_framework.gen_lstm_with("huck_fin_stage2sub",in_stack2,in_first_stack,400)

    #lstm_stage2.train(in_stack2,in_first_stack,20)
    lstm_stage2.save_stateful_predicted(lstm_framework.second_stage_cell_filename,in_stack2)

def run_cumulative_lstm():
    in_base_stack = np.load(lstm_framework.base_input_filename)
    #in_first_stack = np.load(lstm_framework.first_stage_cell_filename)
    in_second_stack = cells_to_outputs(np.load(lstm_framework.second_stage_cell_filename))

    in_stack3 = np.concatenate((in_base_stack,in_second_stack),axis=1)

    lstm_stage3 = lstm_framework.gen_lstm_with("huck_fin_stage3outsub_fixed",in_stack3,in_base_stack,200)

    lstm_stage3.train(in_stack3,in_base_stack,100)
    #[out_vals] = lstm_stage3.state_predict(in_stack3[:3000])
    #print(string_processing.out_list_to_str(out_vals))

#run_level2_lstm()
run_cumulative_lstm()
#run_basic_lstm()
