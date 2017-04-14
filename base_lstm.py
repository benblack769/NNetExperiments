import string_processing
import numpy as np
import lstm
import time


SEQUENCE_LEN = 50

BATCH_SIZE = 128
EPOCS = 100

IN_LEN = string_processing.CHARS_LEN
OUT_LEN = 500
CELL_STATE_LEN = OUT_LEN
HIDDEN_LEN = OUT_LEN + IN_LEN

FULL_OUT_LEN = string_processing.CHARS_LEN


train_str = string_processing.get_str("data/huck_fin.txt")[:10000]
in_vec_list = string_processing.in_vec(train_str)

my_lstm = lstm.LSTM("wide_huck_fin",SEQUENCE_LEN,IN_LEN,OUT_LEN,FULL_OUT_LEN,BATCH_SIZE)

print(train_str)
in_stack = np.vstack(in_vec_list)
instr = np.transpose(np.vstack(in_vec_list))
#print(instr.shape)
def output_trains(num_trains):
    for i in range(num_trains):
        for mid in range(SEQUENCE_LEN,len(train_str)-BATCH_SIZE-1,BATCH_SIZE):
            start = mid - SEQUENCE_LEN
            end = mid + BATCH_SIZE
            in3dtens = np.dstack([instr[:,mid+j:end+j] for j in range(-SEQUENCE_LEN+1,1)] )
            in3dtens = np.rollaxis(in3dtens,-1)
            expected = instr[:,mid+1:end+1]
            yield in3dtens,expected
        print("train_epoc"+str(i),flush=True)
        #for i in range(5): gc.collect()

def train():
    num_trains = 0
    time_last = time.clock()
    train_time = 0
    for inpt,expect in output_trains(EPOCS):
        start = time.clock()
        blocks = my_lstm.train_fn(inpt,expect)
        train_time += time.clock() - start
        output = to_numpys(blocks)
        my_lstm.train_plot_util.update_plots(output)
        my_lstm.shared_value_saver.vals_updated()
        if num_trains%30==0:
            now = time.clock()
            print("train update took {}".format(now-time_last),flush=True)
            print("train time was {}".format(train_time))
            time_last = now
            train_time = 0
        num_trains += 1

    my_lstm.shared_value_saver.force_update()

def to_numpys(outlist):
    return [np.asarray(o) for o in outlist]

def predict():
    pred_text = []
    for inp,_ in output_trains(1):
        outputs = to_numpys(my_lstm.predict_fn(inp))
        my_lstm.predict_plot_util.update_plots(outputs)

        pred_text.append(outputs[0])

    text = [let for t in pred_text for let in np.transpose(t) ]

    outtxt = string_processing.out_list_to_str(text)

    return outtxt

def stateful_predict():
    [outs] = my_lstm.state_predict(in_stack)
    print(outs.shape)
    return string_processing.out_list_to_str(outs)

#print(stateful_predict())
#train()
#print(predict())
print(stateful_predict())
