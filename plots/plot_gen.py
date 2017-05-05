from plot_data import TimeData

foldname = "plot_data/layer1layer5_train_test/"
td = TimeData(foldname+"error_mag.tsv")
td.average_n_steps(50)
td.show_plot()
td = TimeData(foldname+"cell_forget_weights.tsv")
td.show_plot()
td = TimeData(foldname+"cell_add_weights.tsv")
td.show_plot()

'''
fname = "plot_data/cell_time_plot/cell_state_data.tsv"
td = TimeData(fname)
td.crop_window(1000,1120)
td.average_n_steps(5)
td.filter_lines([x for x in range(3,6)])
td.show_plot()
'''
'''huck_fin_stage3outsub_fixed_train_test
#fname ="plot_data/joined_data/"
fname ="plot_data/huck_fin_basic_good_cost2_train_test0/"
# asd
#td = TimeData(fname+"cell_forget_bias.tsv")
#td.filter_lines([x for x in range(2,15)])
#td.show_plot()
td = TimeData(fname+"error_mag.tsv")
#td.average_n_steps(100)
#td.crop_window(100,40000)
td.show_plot()
#td = TimeData(fname+"update_mag.tsv")
#td.average_n_steps(100)
#td.show_plot()
'''


#LSTM bias updating
'''
td = TimeData("plot_data/predict_view00/cell_state.tsv")
td.filter_lines([1,2,3])
td.show_plot()
'''

# noisy cell state generation
'''
td = TimeData("plot_data/predict_view/cell_state.tsv")
td.crop_window(1,100)
td.filter_lines([x for x in range(2,15)])
td.show_plot()
'''

# rmsprop bias plot
'''
td = TimeData("plot_data/rmsprop_test000000000000000/hidbias.tsv")
#td.filter_lines([x for x in range(2,50)])
td.crop_window(0,10000)
td.filter_lines([x for x in range(25,50)])
td.show_plot()
'''


# basic test plot
'''
td = TimeData("plot_data/batch_test0/hidbias.tsv")
td.filter_lines([x for x in range(10,25)])
#td.crop_window(20000,10e50)
#td.average_n_steps(1)
#td.get_diff()
td.show_plot()
'''
