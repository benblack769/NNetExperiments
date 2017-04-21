import string_processing
import numpy as np
train_str = string_processing.get_str("data/huck_fin.txt")
in_vec_list = string_processing.in_vec(train_str)
in_stack = np.vstack(in_vec_list)
np.save("saved_cells/huck_fin_input_data",in_stack)
