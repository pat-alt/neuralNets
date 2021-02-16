# neuralNets

## Prerequisites

This project uses TensorFlow for the implementation of deep neural networks. To install TensorFlow 2 see [here](https://www.tensorflow.org/install/). 

## NLP with deep learning

The [notebook](notebook.ipynb) train various deep learning models to a simple, binary text classification problem. 

### RNN

Recurrent neural networks are based on the idea of persitent thoughts: thinking is modelled as a continuous process that instead of continuously reinventing itself and starting from scratch, evolves gradually and at each step uses information about its prior states. This hierarchical, chain-like nature of RNNs makes them particularly useful for problems that involves sequences, for example, speech recognition or time series analysis. The former is the focus of this small project, so let us dwell on this a little further. Consider the sentence fragment, which is lifted directly from the [data set](data/data.txt) we treat in this project: "[...] will not meet your expectations." Without further context, if you had to classify the sentiment of this text fragment, you would probably label it as negative. When the full sequence of words is revealed the label switches to positive: "Great Subway, in fact it's so good when you come here every other Subway will not meet your expectations." This demonstrates the importance of using prior information that emerges from the context. The sentence also demonstrates that it can be difficult to learn the *context* of a single word just from its nearest neighbours: to understand the role of the word "expectations" in the context of this sentence, it is not enough to look at a few words preceding it. In fact, as we saw above, too small a choice of the context window may lead to wrong conclusions about the sentiment label. In order to account for long-term dependencies we can use a Long Short Term Memory (LSTM) network, a special kind of RNN.

### CNN