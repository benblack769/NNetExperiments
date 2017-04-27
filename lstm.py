import theano
import numpy as np
import theano.tensor as T
import plot_utility
from WeightBias import WeightBias
from shared_save import RememberSharedVals
import os
import time
import lstm_framework
import string_processing

#theano.config.optimizer="fast_compile"
#theano.config.scan.allow_gc=True

class LSTM:
    def __init__(self,save_name,SEQUENCE_LEN,IN_LEN,OUT_LEN,FULL_OUT_LEN,BATCH_SIZE):
        self.SEQUENCE_LEN = SEQUENCE_LEN
        self.BATCH_SIZE = BATCH_SIZE
        CELL_STATE_LEN = OUT_LEN
        HIDDEN_LEN = OUT_LEN + IN_LEN

        inputs = T.tensor3("inputs",dtype="float32")
        stateful_inputs = T.matrix("state_inputs",dtype="float32")
        expected_vec = T.matrix('expected',dtype="float32")

        cell_forget_fn = WeightBias("cell_forget", HIDDEN_LEN, CELL_STATE_LEN,1.0)
        add_barrier_fn = WeightBias("add_barrier", HIDDEN_LEN, CELL_STATE_LEN,0.5)
        add_cell_fn = WeightBias("add_cell", HIDDEN_LEN, CELL_STATE_LEN,0.0)
        to_new_output_fn = WeightBias("to_new_hidden", HIDDEN_LEN, CELL_STATE_LEN,0.5)

        full_output_fn = WeightBias("full_output", CELL_STATE_LEN, FULL_OUT_LEN,0.0)

        self.train_plot_util = train_plot_util = plot_utility.PlotHolder(save_name+"_train_test")
        train_plot_util.add_plot("cell_forget_bias",cell_forget_fn.b,1000)

        self.shared_value_saver = shared_value_saver = RememberSharedVals(save_name+'_lstm')

        weight_biases = (
            cell_forget_fn.wb_list() +
            add_barrier_fn.wb_list() +
            add_cell_fn.wb_list() +
            to_new_output_fn.wb_list() +
            full_output_fn.wb_list()
        )

        def calc_outputs(in_vec,cell_state,out_vec):
            hidden_input = T.concatenate([out_vec,in_vec],axis=0)

            #first stage, forget some info
            cell_mul_val = T.nnet.sigmoid(cell_forget_fn.calc_output(hidden_input))
            forgot_cell_state = cell_state * cell_mul_val

            #second stage, add some new info
            add_barrier = T.nnet.sigmoid(add_barrier_fn.calc_output(hidden_input))
            this_add_val = T.tanh(add_cell_fn.calc_output(hidden_input))
            added_cell_state = forgot_cell_state + this_add_val * add_barrier

            #third stage, get output
            out_all = T.nnet.sigmoid(to_new_output_fn.calc_output(hidden_input))
            new_output = out_all * T.tanh(added_cell_state)

            return added_cell_state,new_output

        def calc_error(expected, actual):
            probs_char = T.exp(actual) / T.sum(T.exp(actual)) # probabilities for next chars
            error = -T.log(T.sum(probs_char*expected)) # softmax (cross-entropy loss#)
            #sqrtdiff = (expected - actual)
            #diff = sqrtdiff * sqrtdiff
            #error = diff.sum()
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
            return all_cell,all_out

        def get_batched_train_pred():
            all_cell_state,all_out = my_scan(inputs)
            out_cell_state = all_cell_state[-1]
            new_out = all_out[-1]

            true_out = T.tanh(full_output_fn.calc_output(new_out))

            error = calc_error(true_out,expected_vec)
            train_plot_util.add_plot("error_mag",error)

            shared_value_saver.add_shared_vals(weight_biases)
            all_grads = T.grad(error,wrt=weight_biases)

            def rms_prop_updates():
                DECAY_RATE = np.float32(0.9)
                LEARNING_RATE = np.float32(0.01)
                STABILIZNG_VAL = np.float32(0.00001)

                gsqr = sum(T.sum(g*g) for g in all_grads)

                grad_sqrd_mag = theano.shared(np.float32(400),"grad_sqrd_mag")
                shared_value_saver.add_shared_val(grad_sqrd_mag)

                grad_sqrd_mag_update = DECAY_RATE * grad_sqrd_mag + (np.float32(1)-DECAY_RATE)*gsqr

                wb_update_mag = LEARNING_RATE / T.sqrt(grad_sqrd_mag_update + STABILIZNG_VAL)
                train_plot_util.add_plot("update_mag",wb_update_mag)

                wb_update = [(wb,wb - wb_update_mag * grad) for wb,grad in zip(weight_biases,all_grads)]
                return wb_update + [(grad_sqrd_mag,grad_sqrd_mag_update)]

            updates = rms_prop_updates()

            # Normal SGD
            # TRAIN_UPDATE_CONST = np.float32(0.1)
            #grads = T.grad(error,wrt=weight_biases)
            #updates = [(sh_var,sh_var - TRAIN_UPDATE_CONST*grad) for sh_var,grad in zip(weight_biases,grads)]

            train_fn = theano.function(
                [inputs,expected_vec],
                train_plot_util.append_plot_outputs([]),
                updates=updates
            )
            return train_fn

        def calc_full_output(inputs,cells,outputs):
            cells,outs = calc_outputs(inputs,cells,outputs)
            full_out = T.tanh(full_output_fn.calc_output(outs))
            return cells,outs,full_out
        def stateful_fn(stateful_ins):
            [cells,outs,full_outs],updates = theano.scan(
                calc_full_output,
                sequences=[stateful_ins],
                outputs_info=[
                    dict(initial=T.zeros((OUT_LEN)),taps=[-1]),
                    dict(initial=T.zeros((OUT_LEN)),taps=[-1]),
                    None
                ],
                n_steps=stateful_inputs.shape[0]
            )
            return cells,outs,full_outs

        def stateful_gen_fn():
            in1 = T.vector("geninput1")
            prevcell = T.vector("prevcell")
            prevout = T.vector("prevout")
            (newcell,newout,full_out) = calc_full_output(in1,prevcell,prevout)
            myfn = theano.function(
                inputs=[in1,prevcell,prevout],
                outputs=[newcell,newout,full_out]
            )
            in_second_stack = np.load(lstm_framework.second_stage_output_filename)
            def get_np(idx):
                arr = np.zeros(string_processing.CHARS_LEN,dtype="float32")
                arr[idx] = 1.0
                return arr
            inidx = 4
            pout = np.zeros(OUT_LEN,dtype="float32")
            pcell = np.zeros(OUT_LEN,dtype="float32")
            all_outs = []
            for x in range(3000):
                inp = np.concatenate((get_np(inidx),in_second_stack[x]))
                pcell,pout,full_out = myfn(inp,pcell,pout)
                outlist = list(full_out)
                inidx = outlist.index(max(outlist))
                #print(inidx)
                all_outs.append(inidx)

            return all_outs
        def get_stateful_predict():
            return theano.function(
                [stateful_inputs],
                [stateful_fn(stateful_inputs)[2]]
            )
        def get_stateful_cells():
            return theano.function(
                [stateful_inputs],
                [stateful_fn(stateful_inputs)[0]]
            )

        def get_stateful_gen_fn():
            return theano.function(
                [],
                [stateful_gen_fn()]
            )

        self.train_fn = get_batched_train_pred()
        self.state_predict = get_stateful_predict()
        self.stateful_cells = get_stateful_cells()
        #self.stateful_gen = stateful_gen_fn()

    def train(self,input_stack,exp_stack,NUM_EPOCS,show_timings=True):
        num_trains = 0
        time_last = time.clock()
        train_time = 0
        for inpt,expect in self.output_trains(np.transpose(input_stack),np.transpose(exp_stack),NUM_EPOCS):
            start = time.clock()
            blocks = self.train_fn(inpt,expect)
            train_time += time.clock() - start
            output = to_numpys(blocks)
            self.train_plot_util.update_plots(output)
            self.shared_value_saver.vals_updated()
            if num_trains%30==0 and show_timings:
                now = time.clock()
                print("train update took {}".format(now-time_last),flush=True)
                print("train time was {}".format(train_time))
                time_last = now
                train_time = 0
            num_trains += 1

        self.shared_value_saver.force_update()
    def plot_stateful_cells(self):
        [cells] = my_lstm.stateful_cells(in_stack[:3000])
        dir_name = "plots/plot_data/cell_time_plot"
        os.makedirs(dir_name, exist_ok=True)
        my_plot = plot_utility.Plot("cell_state_data",None,dir_name)
        for state in cells:
            my_plot.set_update(state)
        my_plot.file.close()
    def save_stateful_cells(self,filename,in_stack):
        [cells] = self.stateful_cells(in_stack)
        np.save(filename,cells)
    def save_stateful_predicted(self,filename,in_stack):
        [cells] = self.state_predict(in_stack)
        np.save(filename,cells)

    def output_trains(self,input_ar,exp_ar,num_trains):
        SEQUENCE_LEN = self.SEQUENCE_LEN
        BATCH_SIZE = self.BATCH_SIZE
        for i in range(num_trains):
            for mid in range(SEQUENCE_LEN,input_ar.shape[1]-BATCH_SIZE-1,BATCH_SIZE):
                start = mid - SEQUENCE_LEN
                end = mid + BATCH_SIZE
                in3dtens = np.dstack([input_ar[:,mid+j:end+j] for j in range(-SEQUENCE_LEN+1,1)] )
                in3dtens = np.rollaxis(in3dtens,-1)
                expected = exp_ar[:,mid+1:end+1]
                yield in3dtens,expected
            print("train_epoc"+str(i),flush=True)

def to_numpys(outlist):
    return [np.asarray(o) for o in outlist]
