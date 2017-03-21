### Using pip and windows

This assumes that you are using pip

First, install numpy

Then install scipy. This requires the BLAS libraries. The best way to do this is by downloading a whl file from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy, and installing the file it using pip, like

    pip install <file>.whl

Then, install theano using pip

You may get a huge error with `error: '::hypot' has not been declared`.

The hackish fix I found is to go into `pyconfig.h` in your python include directory and comment out the lines that look like

    #define hypot _hypot
