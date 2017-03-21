### Installing theano using pip and windows

This assumes that you are using pip

First install scipy and numpy with mkl. This requires the BLAS libraries. The best way to do this is by downloading a whl file, and installing the file it using pip, like

    pip install <file>.whl

Links:

* [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
* [numpy-mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)

Make sure you download the correct version (it will give a nice error if you don't)

Then, install theano using pip

Then try importing theano, and running some basic code. You may get a huge error with `error: '::hypot' has not been declared`.

The hackish fix I found is to go into `pyconfig.h` in your python include directory and comment out the lines that look like

    #define hypot _hypot


This worked for me, please comment if this doesn't work for you.
