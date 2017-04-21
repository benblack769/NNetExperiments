import os
import numpy

plot_dir = "plots/plot_data"

def data_to_bytes(time_frame,numpy_vec):
    flat = numpy_vec.flatten()
    data = (str(val) for val in flat)
    datastr = str(time_frame)+"\t"
    datastr += "\t".join(data)+"\n"
    databytes = bytes(datastr,encoding="utf8")
    return databytes

class Plot:
    def __init__(self,name,data_source,folder_name,skip_updates=0):
        self.data_source = data_source
        self.name = name
        self.update_num = 0
        self.update_mod = 1+skip_updates
        self.time_frame = 1
        filename = os.path.join(folder_name,self.name) + ".tsv"
        unbuffered = 0
        append_opening = "ab"
        self.file = open(filename,append_opening,unbuffered)

    def get_data_graph(self):
        return self.data_source

    def _save_data(self,process_output):
        self.file.write(data_to_bytes(self.time_frame,process_output))

    def set_update(self,process_output):
        if self.update_num % self.update_mod == 0:
            self._save_data(process_output)
        self.update_num += 1
        self.time_frame += 1


class PlotHolder:
    def __init__(self,dir_name,plots = None):
        self.plots = plots if plots != None else []

        self.output_start = None
        self.init_large_dir()
        self.dir_name = os.path.join(plot_dir, dir_name)
        self.init_dir()

    def add_plot(self,name,data_source,skip_updates=0):
        assert self.output_start == None, "add all plots before appending plots!"
        newplot = Plot(name,data_source,self.dir_name,skip_updates)
        self.plots.append(newplot)

    def append_plot_outputs(self,cur_outputs):
        assert self.output_start == None, "only append outputs once!"
        self.output_start = len(cur_outputs)
        plot_outputs = [plot.get_data_graph() for plot in self.plots]
        return cur_outputs + plot_outputs

    def update_plots(self,all_outputs):
        for i,plot in enumerate(self.plots):
            plot.set_update(all_outputs[i+self.output_start])

    def init_large_dir(self):
        os.makedirs(plot_dir, exist_ok=True)

    def init_dir(self):
        while os.path.exists(self.dir_name):
            self.dir_name += "0"

        os.makedirs(self.dir_name)
