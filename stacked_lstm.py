import lstm_framework
import numpy as np
import string_processing

def generate_text_input():
    train_str = string_processing.get_str("data/huck_fin.txt")
    in_vec_list = string_processing.in_vec(train_str)
    in_stack = np.vstack(in_vec_list)
    np.save("saved_cells/huck_fin_input_data",in_stack)

def compare_text(outtxt):
    train_str = string_processing.get_str("data/huck_fin.txt")
    return string_processing.calc_str_errors(outtxt,train_str)
def save_text(filename,outtxt):
    with open(filename,"w") as file:
        file.write(outtxt)

def gen_text():
    in_base_stack = np.load(lstm_framework.base_input_filename)
    #in_first_stack = np.load(lstm_framework.first_stage_cell_filename)
    in_second_stack = np.load(lstm_framework.second_stage_output_filename)

    in_stack3 = np.concatenate((in_base_stack,in_second_stack),axis=1)

    lstm_stage3 = lstm_framework.gen_lstm_with("huck_fin_stage3outsub_fixed",in_stack3,in_base_stack,200)
    all_idxs = lstm_stage3.stateful_gen
    outstr = "".join(string_processing.GOOD_CHARS[idx] for idx in all_idxs)
    print(outstr)
    save_text("sampled_outputs/basic_generated_text.txt",outstr)

def cells_to_outputs(cells):
    #base_constant = 0.6
    #return np.tanh(cells*base_constant)
    sub_cells = cells[1:]-cells[:-1]
    all_cells = np.concatenate((np.zeros((1,cells.shape[1]),dtype="float32"),sub_cells))
    return all_cells

#### Base LSTM training
def run_basic_lstm():
    in_base_stack = np.load(lstm_framework.base_input_filename)
    my_lstm = lstm_framework.gen_lstm_with("huck_fin_basic_good_cost2",in_base_stack,in_base_stack,100)
    #[out_vals] = my_lstm.state_predict(in_base_stack[:3000])
    #outstr = string_processing.out_list_to_str(out_vals)
    #save_text("sampled_outputs/base_output_full.txt",outstr)
    #print(outstr)
    #print(compare_text(outstr))
    my_lstm.train(in_base_stack,in_base_stack,50)
    #my_lstm.save_stateful_cells(first_stage_cell_filename,in_stack)

def run_level2_lstm():
    in_base_stack = np.load(lstm_framework.base_input_filename)
    in_first_stack = cells_to_outputs(np.load(lstm_framework.first_stage_cell_filename))

    in_stack2 = np.concatenate((in_base_stack,in_first_stack),axis=1)

    lstm_stage2 = lstm_framework.gen_lstm_with("huck_fin_stage2sub",in_stack2,in_first_stack,400)

    #lstm_stage2.train(in_stack2,in_first_stack,20)
    lstm_stage2.save_stateful_predicted(lstm_framework.second_stage_output_filename,in_stack2)

def run_cumulative_lstm():
    in_base_stack = np.load(lstm_framework.base_input_filename)
    #in_first_stack = np.load(lstm_framework.first_stage_cell_filename)
    in_second_stack = np.load(lstm_framework.second_stage_output_filename)

    in_stack3 = np.concatenate((in_base_stack,in_second_stack),axis=1)

    lstm_stage3 = lstm_framework.gen_lstm_with("huck_fin_stage3outsub_fixed",in_stack3,in_base_stack,200)

    #lstm_stage3.train(in_stack3,in_base_stack,100)
    [out_vals] = lstm_stage3.state_predict(in_stack3[:3000])
    outstr = string_processing.out_list_to_str(out_vals)
    save_text("sampled_outputs/stacked_output_full.txt",outstr)
    all_idxs = lstm_stage3.stateful_gen
    outstr = "".join(string_processing.GOOD_CHARS[idx] for idx in all_idxs)
    print(outstr)
    save_text("sampled_outputs/basic_generated_text.txt",outstr)

    print(compare_text(outstr))

#save_text("sampled_outputs/proccessed_data.txt",string_processing.get_str("data/huck_fin.txt"))
#np.set_printoptions(edgeitems=40)
#print(np.load(lstm_framework.base_input_filename))
#print(np.load(lstm_framework.second_stage_output_filename))
#print(cells_to_outputs(np.load(lstm_framework.first_stage_cell_filename)))
#run_level2_lstm()
#run_cumulative_lstm()
run_basic_lstm()
#gen_text()
