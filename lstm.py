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

def calc_probs(expected, actual):

    probs = T.nnet.softmax(trans_act)
    lookat_probs = trans_exp * probs
    act_probs = T.sum(lookat_probs,axis=1)
    return probs

def calc_error_catagorized(expected, actual):
    probs = T.nnet.softmax(actual)
    lookat_probs = expected * probs
    act_probs = T.sum(lookat_probs,axis=1)
    stabalizer_val = np.float32(1e-8)
    error = -T.sum(T.log(act_probs + stabalizer_val))
    return error

def error_fn(T_error_funct):
    in1 = T.matrix("arg1")
    in2 = T.matrix("arg2")

    func = theano.function(
        inputs=[in1,in2],
        outputs=[calc_error_catagorized(in1,in2)],
        on_unused_input='warn'
    )
    return func
'''
func2 = theano.function(
inputs=[in1,in2],
outputs=[calc_error_catagorized(in1,in2)]
)
mat1 = (np.array([[1,0,0],[1,0,0],[0,1,0]],dtype="float32"))
mat2 = (np.array([[2,3,2],[0.1,0.01,0.01],[0.1,0.08,0.05]],dtype="float32"))
print(mat1[1:3,0:2])
print(np.sum(mat1,axis=0))
print(np.sum(mat1,axis=1))
print(mat1)
print(mat2)
print(func(mat1,mat2))
print(func2(mat1,mat2))
'''
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

        self.cell_forget_fn = WeightBias(save_name+"_cell_forget", self.HIDDEN_LEN, self.CELL_STATE_LEN)
        self.add_barrier_fn = WeightBias(save_name+"_add_barrier", self.HIDDEN_LEN, self.CELL_STATE_LEN)
        self.add_cell_fn = WeightBias(save_name+"_add_cell", self.HIDDEN_LEN, self.CELL_STATE_LEN)
        self.to_new_output_fn = WeightBias(save_name+"_to_new_hidden", self.HIDDEN_LEN, self.CELL_STATE_LEN)

    def get_weight_biases(self):
        return (
            self.cell_forget_fn.wb_list() +
            self.add_barrier_fn.wb_list() +
            self.add_cell_fn.wb_list() +
            self.to_new_output_fn.wb_list()
        )

    def set_train_watch(self,train_plot_util):
        train_plot_util.add_plot(self.save_name+"_cell_forget_weights",self.cell_forget_fn.W,100,300)
        train_plot_util.add_plot(self.save_name+"_cell_add_weights",self.add_cell_fn.W,100,300)

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

test_layer = LSTM_Layer("test",4,6)
inpu = T.matrix("Arg",dtype="float32")
cell_state = T.matrix("Arg2",dtype="float32")
outstart = np.zeros((10,5))
outstart = np.zeros((10,5))
class TanhLayer:
    def __init__(self,save_name,IN_LEN,OUT_LEN):
        self.save_name = save_name
        self.IN_LEN = IN_LEN
        self.OUT_LEN = OUT_LEN
        self.layer_fn = WeightBias(save_name+"_tanh_wb", IN_LEN, OUT_LEN)
    def get_weight_biases(self):
        return self.layer_fn.wb_list()

    def set_train_watch(self,train_plot_util):
        train_plot_util.add_plot(self.save_name+"tanh_fn",self.layer_fn.W,100,300)

    def calc_output(self,inputs,_no_cells):
        out = T.tanh(self.layer_fn.calc_output(inputs))
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

        self.IN_LEN = layer1.IN_LEN
        self.OUT_LEN = layer2.OUT_LEN

        self.save_name = layer1.save_name + layer2.save_name

    def get_weight_biases(self):
        return self.layer1.get_weight_biases() + self.layer2.get_weight_biases()

    def calc_output(self,inputs,cells):
        prevcell1,prevcell2 = cells
        out1,outcell1 = self.layer1.calc_output(inputs,prevcell1)
        out2,outcell2 = self.layer2.calc_output(out1,prevcell2)
        return out2,(outcell1+outcell2)

    def set_train_watch(self,train_plot_util):
        self.layer1.set_train_watch(train_plot_util)
        self.layer2.set_train_watch(train_plot_util)

    def init_cells(self):
        return [self.layer1.init_cells(), self.layer2.init_cells()]

    def init_cells_batched(self,batch_size):
        return [self.layer1.init_cells_batched(batch_size),self.layer2.init_cells_batched(batch_size)]

def build_list_in_pattern_help(initer,pattern):
    if not isinstance(pattern,list):
        return next(initer)
    outlist = []
    for l in pattern:
        outlist.append(build_list_in_pattern_help(initer,l))
    return outlist
def build_list_in_pattern(inlist,pattern):
    return build_list_in_pattern_help(iter(inlist),pattern)
def flatten(deeplist):
    if isinstance(deeplist,list) or isinstance(deeplist,tuple):
        for l in deeplist:
            for x in flatten(l):
                yield(x)
    else:
        yield (deeplist)
def output_nps_fn(train_fn):
    def newfn(*args):
        blocks = train_fn(*args)
        outputs = to_numpys(blocks)
        return blocks
    return newfn

class RMSpropOpt:
    def __init__(self,LEARNING_RATE,DECAY_RATE=0.9):
        self.LEARNING_RATE = np.float32(LEARNING_RATE)
        self.DECAY_RATE = np.float32(DECAY_RATE)
        self.grad_sqrd_mag = theano.shared(np.float32(400),"grad_sqrd_mag")

    def get_shared_states(self):
        return [self.grad_sqrd_mag]

    def updates(self,error,weight_biases):
        STABILIZNG_VAL = np.float32(0.0001**2)
        all_grads = T.grad(error,wrt=weight_biases)

        gsqr = sum(T.sum(g*g) for g in all_grads)

        grad_sqrd_mag_update = self.DECAY_RATE * self.grad_sqrd_mag + (np.float32(1)-self.DECAY_RATE)*gsqr

        wb_update_mag = self.LEARNING_RATE / T.sqrt(grad_sqrd_mag_update + STABILIZNG_VAL)

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
        self.BATCH_SIZE = BATCH_SIZE
        self.SEQUENCE_LEN = SEQUENCE_LEN
        self.forward_prop = forward_prop
        self.optimizer = optimizer
        self.cost_fn = cost_fn
        self.save_name = self.forward_prop.save_name
        self.shared_value_saver = RememberSharedVals(self.save_name+'_lstm')

        self.shared_value_saver.add_shared_vals(forward_prop.get_weight_biases())
        self.shared_value_saver.add_shared_vals(optimizer.get_shared_states())

    def my_scan(self,stateful_ins,use_batching):
        if use_batching:
            init_cells = self.forward_prop.init_cells_batched(self.BATCH_SIZE)
        else:
            init_cells = self.forward_prop.init_cells()

        out_inf_cells = [dict(initial=icell,taps=[-1]) for icell in flatten(init_cells)]

        def calc_output_wrapper(*args):
            pattenred_args = build_list_in_pattern(args[1:],init_cells)
            out, cells = self.forward_prop.calc_output(args[0],pattenred_args)
            print(out)
            return [out]+cells

        all_outputs,updates = theano.scan(
            calc_output_wrapper,
            sequences = [stateful_ins],
            outputs_info = [None]+out_inf_cells
        )
        outs = all_outputs[0]
        cells=all_outputs[1:]
        return cells,outs

    def prop_through_sequence(self,invecs):
        cells,outs = self.my_scan(invecs,True)
        return outs[-1]

    def get_batched_train_pred(self):
        inputs = T.tensor3(self.save_name+"_inputs",dtype="float32")
        expected = T.matrix(self.save_name+'_expected',dtype="float32")

        train_plot_util = plot_utility.PlotHolder(self.save_name+"_train_test")
        self.forward_prop.set_train_watch(train_plot_util)

        full_output = self.prop_through_sequence(inputs)

        error = self.cost_fn(expected,full_output)
        train_plot_util.add_plot("error_mag",error)

        updates = self.optimizer.updates(error,self.forward_prop.get_weight_biases())

        train_fn = theano.function(
            [inputs,expected],
            train_plot_util.append_plot_outputs([]),
            updates=updates
        )
        np_output_fn = output_nps_fn(train_fn)
        train_fn_plotup = train_plot_util.get_plot_update_fn(np_output_fn)
        train_fn_saveup = self.shared_value_saver.share_save_fn(train_fn_plotup)
        return train_fn_saveup

    def stateful_gen_fn():
        #invalid function
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

    def get_stateful_predict(self):
        stateful_inputs = T.matrix(self.save_name+"_state_inputs",dtype="float32")
        cells,outs = self.my_scan(stateful_inputs,False)
        return theano.function(
            [stateful_inputs],
            [outs]
        )
    def get_stateful_cells(self):
        stateful_inputs = T.matrix(self.save_name+"_state_inputs",dtype="float32")
        cells,outs = self.my_scan(stateful_inputs,False)
        return theano.function(
            [stateful_inputs],
            [cells]
        )

def to_numpys(outlist):
    return [np.asarray(o) for o in outlist]

def train(learner,input_stack,exp_stack,NUM_EPOCS,show_timings=True):
    num_trains = 0
    time_last = time.clock()
    train_time = 0
    train_fn = learner.get_batched_train_pred()
    for inpt,expect in output_trains(learner,np.transpose(input_stack),np.transpose(exp_stack),NUM_EPOCS):
        start = time.clock()
        output = train_fn(inpt,expect)
        train_time += time.clock() - start
        if num_trains%30==0 and show_timings:
            now = time.clock()
            print("train update took {}".format(now-time_last),flush=True)
            print("train time was {}".format(train_time))
            time_last = now
            train_time = 0
        num_trains += 1

    learner.shared_value_saver.force_update()

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

def output_trains(learner,input_ar,exp_ar,num_trains):
    SEQUENCE_LEN = learner.SEQUENCE_LEN
    BATCH_SIZE = learner.BATCH_SIZE
    for i in range(num_trains):
        for mid in range(SEQUENCE_LEN,input_ar.shape[1]-BATCH_SIZE-1,BATCH_SIZE):
            start = mid - SEQUENCE_LEN
            end = mid + BATCH_SIZE
            in3dtens = np.dstack([input_ar[:,mid+j:end+j] for j in range(-SEQUENCE_LEN+1,1)] )
            in3dtens = np.rollaxis(in3dtens,-1)
            expected = exp_ar[:,mid+1:end+1]
            yield in3dtens,expected
        if num_trains > 1:
            print("train_epoc"+str(i),flush=True)
