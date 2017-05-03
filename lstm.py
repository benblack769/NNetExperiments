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
import itertools

#theano.config.optimizer="fast_compile"
#theano.config.scan.allow_gc=True

def calc_error_catagorized(expected, actual):
    probs_not_normalized = T.exp(actual)
    norm_probs = probs_not_normalized / T.sum(probs_not_normalized) # probabilities for next chars
    error = -T.sum(T.log(norm_probs)*expected) # softmax (cross-entropy loss#)
    return error

def calc_error_squared(expected, actual):
    sqrtdiff = (expected - actual)
    diff = sqrtdiff * sqrtdiff
    error = diff.sum()
    return error

class LSTM_Layer:
    def __init__(self,save_name,IN_LEN,OUT_LEN):
        self.save_name = save_name
        self.IN_LEN = IN_LEN
        self.OUT_LEN = OUT_LEN
        self.CELL_STATE_LEN = OUT_LEN
        self.HIDDEN_LEN = OUT_LEN + IN_LEN

        self.cell_forget_fn = WeightBias(save_name+"_cell_forget", self.HIDDEN_LEN, self.CELL_STATE_LEN,1.0)
        self.add_barrier_fn = WeightBias(save_name+"_add_barrier", self.HIDDEN_LEN, self.CELL_STATE_LEN,0.5)
        self.add_cell_fn = WeightBias(save_name+"_add_cell", self.HIDDEN_LEN, self.CELL_STATE_LEN,0.0)
        self.to_new_output_fn = WeightBias(save_name+"_to_new_hidden", self.HIDDEN_LEN, self.CELL_STATE_LEN,0.5)

    def get_weight_biases(self):
        return (
            self.cell_forget_fn.wb_list() +
            self.add_barrier_fn.wb_list() +
            self.add_cell_fn.wb_list() +
            self.to_new_output_fn.wb_list()
        )

    def set_train_watch(self,train_plot_util):
        train_plot_util.add_plot("cell_forget_weights",self.cell_forget_fn.W,1000,300)
        train_plot_util.add_plot("cell_add_weights",self.add_cell_fn.W,1000,300)

    def calc_output(self,in_vec,cell_states):
        '''
        inherited classes can use:
        calculates cell state, output, from inputs
        '''
        [cell_state,out_vec] = cell_states

        hidden_input = T.concatenate([out_vec,in_vec],axis=0)

        #first stage, forget some info
        cell_mul_val = T.nnet.sigmoid(self.cell_forget_fn.calc_output(hidden_input))
        forgot_cell_state = cell_state * cell_mul_val

        #second stage, add some new info
        add_barrier = T.nnet.sigmoid(self.add_barrier_fn.calc_output(hidden_input))
        this_add_val = T.tanh(self.add_cell_fn.calc_output(hidden_input))
        added_cell_state = forgot_cell_state + this_add_val * add_barrier

        #third stage, get output
        out_all = T.nnet.sigmoid(self.to_new_output_fn.calc_output(hidden_input))
        new_output = out_all * T.tanh(added_cell_state)

        return new_output,[added_cell_state,new_output]

    def init_cells_batched(self, batch_size):
        return [T.zeros((self.CELL_STATE_LEN,batch_size)),T.zeros((self.OUT_LEN,batch_size))]

    def init_cells(self):
        return [T.zeros(self.CELL_STATE_LEN),T.zeros(self.OUT_LEN)]

class TanhLayer:
    def __init__(self,save_name,IN_LEN,OUT_LEN):
        self.save_name = save_name
        self.IN_LEN = IN_LEN
        self.OUT_LEN = OUT_LEN
        self.layer_fn = WeightBias(save_name+"_tanh_wb", IN_LEN, OUT_LEN)
    def get_weight_biases(self):
        return self.layer_fn.wb_list()

    def set_train_watch(self,train_plot_util):
        return

    def calc_output(self,inputs,_no_cells):
        out = T.tanh(self.layer_fn.calc_output(outs))
        return out,[]

    def init_cells_batched(self, batch_size):
        return []
    def init_cells(self):
        return []


class TwoLayerLSTM:
    def __init__(self,layer1,layer2):
        assert layer1.OUT_LEN == layer2.IN_LEN, "layers do not match up"
        self.layer1 = layer1
        self.layer2 = layer2

    def get_weight_biases(self):
        return self.layer1.get_weight_biases() + self.layer2.get_weight_biases()

    def calc_output(self,inputs,cells):
        prevcell1,prevcell2 = cells
        out1,outcell1 = self.layer1.calc_output(inputs,prevcell1)
        out2,outcell2 = self.layer2.calc_output(out1,prevcell2)
        return out2,[outcell1,outcell2]

    def set_train_watch(self,train_plot_util):
        self.layer1.set_train_watch(train_plot_util)
        self.layer2.set_train_watch(train_plot_util)

    def init_cells(self):
        return [self.layer1.init_cells(), self.layer2.init_cells()]

    def init_cells_batched(self):
        return [self.layer1.init_cells_batched(),self.layer2.init_cells_batched()]

def build_list_in_pattern_help(initer,pattern):
    if not isinstance(pattern,list):
        return next(initer)
    outlist = []
    for l in pattern:
        outlist.append(build_list_in_pattern_help(initer,pattern))
    return outlist
def build_list_in_pattern(inlist,pattern):
    return build_list_in_pattern_help(iter(inlist),pattern)
def flatten_help(build_list,deeplist):
    if isinstance(l,list):
        for l in deeplist:
            flatten_help(build_list,deeplist)
    else:
        build_list.append(deeplist)
def flatten(deeplist):
    out = []
    flatten_help(out,deeplist)
    return out

class RMSpropOpt:
    def __init__(self):
        self.grad_sqrd_mag = theano.shared(np.float32(400),self.save_name+"_grad_sqrd_mag")

    def get_shared_states(self):
        return [self.grad_sqrd_mag]

    def updates(self,error,weight_biases):
        DECAY_RATE = np.float32(0.9)
        LEARNING_RATE = np.float32(0.01)
        STABILIZNG_VAL = np.float32(0.00001)

        all_grads = T.grad(error,wrt=weight_biases)

        gsqr = sum(T.sum(g*g) for g in all_grads)

        grad_sqrd_mag_update = DECAY_RATE * self.grad_sqrd_mag + (np.float32(1)-DECAY_RATE)*gsqr

        wb_update_mag = LEARNING_RATE / T.sqrt(grad_sqrd_mag_update + STABILIZNG_VAL)
        train_plot_util.add_plot("update_mag",wb_update_mag)

        wb_update = [(wb,wb - wb_update_mag * grad) for wb,grad in zip(weight_biases,all_grads)]
        return wb_update + [(self.grad_sqrd_mag,grad_sqrd_mag_update)]

class SGD_Opt:
    def get_shared_states(self):
        return []
    def updates(self,error,weight_biases):
        TRAIN_UPDATE_CONST = np.float32(0.1)
        grads = T.grad(error,wrt=weight_biases)
        updates = [(sh_var,sh_var - TRAIN_UPDATE_CONST*grad) for sh_var,grad in zip(weight_biases,grads)]
        return updates

class Learner:
    def __init__(self,forward_prop,optimizer,cost_fn,BATCH_SIZE,SEQUENCE_LEN):
        self.forward_prop = forward_prop
        self.optimizer = optimizer
        self.cost_fn = cost_fn
        self.save_name = self.forward_prop.save_name
        self.shared_value_saver = RememberSharedVals(forward_prop.save_name+'_lstm')

        self.shared_value_saver.add_shared_vals(forward_prop.get_weight_biases())
        self.shared_value_saver.add_shared_vals(optimizer.get_shared_states())

    def my_scan(self,state_ins,use_batching):
        if use_batching:
            init_cells = self.forward_prop.init_cells_batched()
        else:
            init_cells = self.forward_prop.init_cells()

        out_inf_cells = [dict(initial=icell,taps=[-1]) for icell in flatten(init_cells)]
        out_info_all = list(itertools.chain((None,oi_cell) for oi_cell in out_inf_cells))

        def calc_output_wrapper(*args):
            pattenred_args = build_list_in_pattern(args,init_cells)
            return flatten(self.forward_prop.calc_output(pattenred_args))

        [cells,outs],updates = theano.scan(
            calc_output_wrapper,
            sequences = [stateful_ins],
            outputs_info = out_info_all
        )
        return cells,outs

    def stateful_output(stateful_ins):
        cells,outs = self.my_scan(stateful_ins,False)
        return outs

    def prop_through_sequence(self,invecs):
        cells,outs = self.my_scan(stateful_ins,True)
        return outs[-1]

    def get_batched_train_pred(self):
        inputs = T.tensor3(self.save_name+"_inputs",dtype="float32")
        expected = T.matrix(self.save_name+'_expected',dtype="float32")

        train_plot_util = plot_utility.PlotHolder(self.save_name+"_train_test")
        self.forward_prop.set_train_watch(train_plot_util)

        full_output = self.prop_through_sequence(inputs)

        error = self.cost_fn(full_output,expected)
        train_plot_util.add_plot("error_mag",error)

        updates = self.optimizer.updates(error,self.forward_prop.get_weight_biases())

        train_fn = theano.function(
            [inputs,expected_vec],
            train_plot_util.append_plot_outputs([]),
            updates=updates
        )
        return train_fn

    '''def stateful_gen_fn():
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

        return all_outs'''

    def get_stateful_predict(self):
        stateful_inputs = T.matrix(self.save_name+"_state_inputs",dtype="float32")
        return theano.function(
            [stateful_inputs],
            [self.stateful_fn(stateful_inputs)[2]]
        )
    def get_stateful_cells():
        stateful_inputs = T.matrix(self.save_name+"_state_inputs",dtype="float32")
        return theano.function(
            [stateful_inputs],
            [self.stateful_fn(stateful_inputs)[0]]
        )

def to_numpys(outlist):
    return [np.asarray(o) for o in outlist]

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
