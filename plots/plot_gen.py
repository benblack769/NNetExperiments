from plot_data import TimeData

# asd
td = TimeData("plot_data/train_test/cell_forget_bias.tsv")
td.show_plot()
td = TimeData("plot_data/train_test/error_mag.tsv")
td.average_n_steps(100)
td.show_plot()
td = TimeData("plot_data/train_test/update_mag.tsv")
td.average_n_steps(100)
td.show_plot()



#LSTM bias updating
'''
td = TimeData("plot_data/train_test000/cell_forget_bias.tsv")
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
