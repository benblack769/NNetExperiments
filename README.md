# Creating long term memory for a neural network

## Introduction

### Assumed Knowledge

Now, to start off, I am no expert in neural nets. I spent a month or two learning about them from excellent sources like [this](http://neuralnetworksanddeeplearning.com/).

If you have a good understanding of linear algebra and multidimensional calculus, you should be able to pick it up quite quickly. But neural nets do not use super advanced mathematical concepts, so don't let the math scare you too much! You should be able to learn the math alongside the network specific info. Why am I going over this? Because for this post, I will assume some basic knowledge about neural nets. There are already a million sources to learn the basics, and far too few going over more advanced concepts.

### Inspiration and Goal

UNDERSTANDING IS THE MOST IMPORTANT THING FOR THIS FINAL PROJECT

STRUCTURE LEADS TO MORE SIGNIFICANT RESULTS THAN MATHMATICAL TRICKS.

THE LEARNING TASK OF GUESSING THE NEXT LETTER IS NOT REALLY A RENFORCEMENT LEARNING TASK. RATHER, i

THE COMPRESSION ALGORIHM IS USEFUL BECAUSE IF YOU CAN DO THAT< THEN YOU CAN DO ALL THESE OTHER THINGS TOO, like natural language processing.

To start out, I will cover LSTMs, as they are the base for what I am trying to do. There is an awesome description of them [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), and I will be using diagrams from that source (which are generously published as public domain).

One thing to know about LSTMs is that they are invented in 1999, and are still by far the best general purpose time series predictor known. This is not due to a lack of effort. There have been advances in

* Mathematics: Hessian free optimization, an  effective second order optimization model, has been successfully applied to neural networks)
* Neuroscience: We have a much better idea of how neurons actually interact)
Recurrent neural network design: Inspired by LSTMs, many people have created similar models which they though would solve some problems that LSTMs seem to have. None of the dozens of these are better, although some are marginally simpler. A meta-study done recently found that some models are marginally better.
* Augmented networks: Many people think the next big thing in neural nets are ones given external memory to work with. These are often combined with LSTM networks on some of the most important pattern recognition problems. Google's new translation algorithm uses Attention networks to augment LSTMs.

So what makes these so powerful? What made people think they could be improved with minor changes, and why were they wrong? What can't it do? If we ever want to make something even better, perhaps solving some of the problems with LSTMs, we need to be able to answer these questions. I think that LSTMs might encode something fundamental about intelligence, although it may not be easy to see what that is.

### Recurrent Networks and Time Sequence Learning

First, what is a recurrent network? It tries to solve a common problem that humans solve, which is recognizing sequences over time. It does this by looking at the data one timestep at a time, only storing a fixed ammount of data from previous timesteps. For example, RNNs are used for voice recognition, i.e. turning a sequence of sound waves into a sequence of letters.

A recurrent network broadly it looks like this:

![long term dependencies image](https://raw.githubusercontent.com/weepingwillowben/music_net/master/diagrams/RNN-longtermdependencies.png "Long term dependencies")

The major parts are the input, a state which is a function of the previous inputs, and the output, which is a function of the previous state. For voice recognition, the input would be some representation of sound waves, and the output would be text. In order to train the model, the output would then be compared to the expected output, which is usually computed by hand (by writing out the text that the person is saying). Then, as long as the function A is differentiable, we can use backpropagation with gradient descent in order to get it to learn. I'll explore gradient descent in more detail later.

So how do we construct such a function so that we can nicely change the parameters like that? The obvious construction follows directly from ordinary deep neural networks.

#### Description of ordinary recurrent networks

Traditional RNNs are built as follows:

![long term dependencies image](https://raw.githubusercontent.com/weepingwillowben/music_net/master/diagrams/LSTM3-SimpleRNN.png "Long term dependencies")

The

#### Why ordinary RNNs fail
cite http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf
#### Description of LSTM networks
#### Some broad intuition for why they might work better

#### Gradient descent

We store the inputs and states for several time steps back. Then we attempt to find out which internal parameters in the function contributed to the output the most. We then change these parameters so that they make the function output what we expected it to. To help conceptualize this, look at the diagram above. Suppose h3 is purely a function of x0 and x1. Then we would want to change the parameters in the A function so that this occurs.

## Actual Implementations and Some Results

### First Neural Net

To get back into practice with neural nets, and accustomed myself to the working of Theano, I made an extremely simple 3 layer neural net, and trained it on an extremely simple learning task. The task was to learn a Caesar cipher of 1, that is the letters A,B,C,...,Y,Z turns into B,C,D,...,Z,A. Even though this is a trivial learning task, it reveals some interesting facts about neural networks and gradient descent.

The data set I was training on is an excerpt from Douglas Adam's *Hitchhiker's Guide to the Galaxy*. It is in `data/test_text.txt`. The code to run the learning scheme is in `theanoexp.py`. It encodes the letters with a 26 node input and output, which correspond to the 26 letters, and has a hidden layer size of 100.

#### Random initialization bug

When I first wrote it up, I had the problem where it would always output the same random letter, no matter what training occurred. I eventually figured out that the random weights and biases I initialized the matrix with were much too large. Just an example of one of the ways ANNs depend on values to work, not control flow.

#### Unstable Biases

I used my tool to get the biases as they were training to see if they eventually stabilized. The following records a selection of the hidden layer biases 300 cycles through the text.

![alt](https://raw.githubusercontent.com/weepingwillowben/music_net/master/plots/basic_test_plots/selected_hidden_biases.png "biases in output layer")

As you can see, it does not seem to stabilize perfectly, even over hundreds of iterations over the dataset. Instead it seems to slip every once in awhile into a new sort of state.

## Optimizers (RMSprop)

First, a quick overview of the history of ANN optimizers:

http://sebastianruder.com/optimizing-gradient-descent/index.html#adadelta

Gradient descent has many well studied mathematical problems. It gets stuck in local minima, it can bounce around between the sides of a valley before coming to a rest at the bottom, and other problems. Attempts to come to a solution have often been too slow, or not generally applicable. The evolution of these has accumulated in three general purpose optimizing algorithms, Adam, AdaDelta, and RMSprop. All of these have slight alterations to the basic gradient descent which try to intelligently pick the size of the step.  I implemented RMSprop because of its simplicity and power. Amazingly, this widely used and generally admired optimizer was first introduced to the world through some slides on a Coursera class ([link](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)).

#### Description

Here is the actual formula that is used to calculate it:

![rmsprop formula](https://raw.githubusercontent.com/weepingwillowben/music_net/master/diagrams/rmsprop.PNG "Long term dependencies")

Gramma and alpha are parameters that you get to pick. gamma is usuaally around 0.9. Theta is a measure of the magnitudes of the preious gradients. In theory, what this does is make the learning rate smaller when the gradient is steeper, perhaps slowly lowering the values into a "canyon" (where normal SGD will overshoot the valley of minimal loss and end up on the other cliff). It will also make larger changes when the gradient is smaller, which can help it get out of local minima.

[This](http://sebastianruder.com/optimizing-gradient-descent/) great blog describing this problem in detail buit some small dimentional visualizations to capture this intuition, and show how it performs in comparison to other approaches, like the momentum approach.

This one shows an example of how momentum approaches can do worse than RMSprop and AdaDelta.

![grad1](https://raw.githubusercontent.com/weepingwillowben/music_net/master/diagrams/grad1.gif "Long term dependencies")

This one shows how RMSprop can get out of an unstable equilibria very quickly.

![grad2](https://raw.githubusercontent.com/weepingwillowben/music_net/master/diagrams/grad2.gif "Long term dependencies")

However, these are contrived low dimentional problems. We want to know if these things actually work on extremely high dimentional neural networks training on real problems.

In order to see if this stabilized anything, I went ahead and ran it, capturing the updating biases as it updated. Ignore the x axis, this chart is on the same sort of timescale as the one above using ordinary SGD.

![alt](https://raw.githubusercontent.com/weepingwillowben/music_net/master/plots/basic_test_plots/rmprop_long_update.png "biases in output layer using rmsprop")

You can see that at the very beginning, it is changing much slower than SGD. This is almost certainly the primary intention of RMSprop: to stablize outcomes by forcing it to change slower so it doesn't jump to some horrible state when updating. However, it is not so clear that it always works better since sometimes you really do just learn the easy stuff to learn as quickly as possible.

Since this learning task is so easy, I expect it to actually performs worse on this learning task (or at the very least not much better). This is because the space is so simple, it is OK for SGD to update quickly at the beginning. So all rmsprop does is slow it down. The space it is optimizing over is not nearly complex enough to need it. However, I did not have enough time to verify this using real data.

The fact that in actual learning tasks, RMSprop and similar methods are so popular seem to indicate that in real learning tasks, steady learning processes are essential for gradient based learning to work.

We do not think of ourselves learning steadily, though. When we see something suprizing, something wildly differnt from the norm, we often focus on it, and think about it a lot. RMSprop ensures the opposite, that when something is encountered that is different from the norm, the total ammount the ANN learns from it is in fact not much greater than what we expect to see.

To me, this suggests that RMSprop is simply operating at a different level than what we think of learning in humans.

Instead, it seems to be more similar to how humans experience sensation. When we have more inputs: our eyes are open, we are seeing lots of bright things, listening to loud music, etc, then it is harder to focus on smaller details. Only the most extreme sensations stand out.

Sensation and learning do not really seem to be innately related. In fact, they are in a way fundamentally differnet: rmsprop's optimizer has to do with regularizing outputs, and sensation has to do with managing inputs. h I think RMSprop and the human adjustment of senation are solving a similar problem: that just because there is more to see, or more to learn does not necessarily mean that it is that much more important.

## LSTM

### Structure

Colah's blog post [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is a great description of the structural aspects of LSTMs.

![lstm-diagram](https://raw.githubusercontent.com/weepingwillowben/music_net/master/plots/basic_test_plots/LSTM3-chain.png "biases in output layer using rmsprop")



### Learning task

I will start out with a rather simple learning task: recognize most likely next letter.

For example, if it gets input "I am goin", it should output "g", as it is pretty obvious that the statement is: "I am going".

One nice property of this task is that if you know English, you should be able to perform the task better. So perhaps in order to perform the task, it also will learn a bit about English in the process.

#### Training text

I chose the text huckleberry fin, partly because its copyright is expired, and partly because it has a distinctive writing style that we can see if the network identifies. I found a text copy on Project Gutenberg ([license](
https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License))

### Initial results



## Part 2: improving the network

Unfortunately, it is difficult to reasonably test theoretical hypothesis of how algorithms work. We know they work, and we can state properties of them, but especially something as complicated as LSTMs, it is hard to concretely track down any sort of causation or fundamental nature.

However, there is some intuition associated with it.

One way I like doing is to alter the algorithm based on

Because solving problems is easier than figuring out fundamental reasons for phenomena, I also want to explore how one might improve LSTMs. Attention based networks have made significant strides in time sequence analysis, such as translation. But current methods are inefficient for long time series. One potential goal of a lower level network model is to condense the important information in a long time series into a relatively small number of steps, each of which containing a reasonable amount of information. In other words, to compress the time scale of the data, without hugely increasing the spatial scale. It is not clear that current techniques like stacking LSTMS do this effectively.

I think that this condensation over time is vital to creating human level intelligence. The very idea of attention is related to the conscious self, which can only focus on a small number of things at a time. So it relies on something lower level to condense the relevant information to something we can reasonably expect the machine to focus on.

### Model 1



### Model 2

Model 1 lacked some basic intutive framework, which is why it is so hard to interpret its results. So I designed a new model.

It is easy to imageing music as a set of waves. Not the individual sound waves, but the overall feeling of the rise and fall of the music, and all the little twists inside that. Perhaps text works in a similar sort of way. There are rises and falls of sound Especially considering poetry and vocal music, perhaps it is not completly implausible to think of text in this way.  working the same sort of way, with a

## Appendix A: Thoughts on Debugging Neural Networks

Now, when people show you their neural network code, you go off, and try to run it, and it may work, but good luck making any changes. Neural nets are notoriously tricky. Out of all the code I have ever written, they have the greatest barriers to debugging. The code will run with several major errors, but it will not do what you want to do without everything being perfect. This is the case because there are no conditionals in the main part of the code. You can write out just about every ANN ever invented without loops, if statements or recursion. This is not only speculation or theory, some code I wrote does just this, outputting code without loops that computes a simple, but practically useful neural net (see my [Graph Optimizer](https://github.com/weepingwillowben/GraphOptimizer) project for more details). What this means is that every part of the code is important to every part of the computation. There is not any part of the code which is not run for certain problems.

I this is an interesting contrast to standard AI techniques, which may be fiddly, but if you make one minor error in the code, then usually most things work fine, or it crashes immediately, or at the very least, it is easy to trace back what exactly went wrong. In general, it is easy to debug. This is because for GOFAI, we are navigating around some sort of finite state machine, or some sort of symbol processor, or decision tree, or something, and in all of these, we can think about the program moving around, doing different things in different cases. This is the case because the symbolic concepts of the AI tend to be directly correspond to exact lines of code, or at least a fairly small section of code.

I belive that this difference between GOFAI and neural networks is intracatly interrelated to the difference between symbolic and sub-symbolic computation. This suggests that while sub-symbolic computation poses many solutions to philosophical and computational problems, it poses an engeneering problem: how do we examine a bit of code which every part, every peice of data affects every other part?

Solving this problem requires new debugging tools, new tests, new debugging thought processes, and maybe even new hardware. Instead of thinking about an instruction pointer going around, we need to think about data changing, its interactions with data around it, how it accumulates, increments, and compounds. We need visualizations which allows us to see how the data is changing, not which instruction the computer is following.

My solution was to create line graphs that displayed certain values over time. Many of those graphs are above. In order to do this easily I made a simple tool that save values in the network to a file (plot_utility.py), and a tool which uses MatPlotlib to create a visualization of that data (plot_data.py).

However, this is admitedly a deeply imperfect solution, as these line graphs only display a small fraction of the data, and not too clearly at that. It is also extremely difficult to see how they are intereacting. Causality is completly confounded. This makes my plots not as useful as I hoped they would be. However, I belive that there are better ways out there to look at the real computation happening inside the data.


## Appendix B: Execution and Debugging of Neural Network Code

### Theano

Ok, now on to creating the code. I chose Theano, which I think is a really cool framework, independent of all the cool neural networks it allows you to build. Basically, it view the program as a graph. This allows the following:

* Manipulate the program graph in order to make the code faster.
    * memory reuse optimization
* Hardware support
    * Very easy and efficient GPU computation
    * cluster computation, including muti-GPU computation
* Allows for easy generation of code flow charts
* Automatic and efficient computation of gradients, saving a lot of code

Unfortunately, it comes with the downside that the code creation and execution are more separated than usual, making it somewhat harder to debug. Luckily, the Theano developers put a lot of work into making this problem better.

### Saving/loading of trained weights

Training takes awhile. At least when I did it, I wanted to stop the process so that I could view plots, adjust learning rates, and see if it is actually learning. I then wanted to start it up again right where it left off. The `shared_save.py` file is dedicated to this task.

I found this absolutely essential to effectively working with this experimental work.

It also was nice because then I could use Amazon Spot GPU instances, which are much cheaper than On Demand instances, but can interrupt your code at will.

### Batch Learning

The primary hardware optimization for Neural Networks is batched learning. Some people mistakenly think that this has some deeper meaning which it does not (batches of size 1 are optimal for sequential processors), but it does allow significant data parallelism. It makes the difference between GPUs being useful and useless, and even on CPUs makes it many times faster. I used batch sizes of 128 or 256 on GPUs, which allowed my model to train dozens of times faster.

### GPU install process

It turns out that installing all the software for GPU computation using Theano is amazingly hard, at every level. CUDA, which Nvidia puts substantial amount of effort into making installation easy, requires the unbelievably bloated VS 14 compiler on windows, and substantial system configuration on Linux. Theano's gpuarray backend is broken on windows and you need to install from source on Linux.

Basically, if all this installation process was easier, people would use GPUs much more for computation, and probably deep learning as a field of study would be more accessible.
