# Discovering AutoKeras Search Space

## Introduction

Using [AutoKeras](https://autokeras.com/) for image classification, I found that on different runs I got very different results.
I had expected that results would converge over a larger number of trials, but they didn't. I wanted to know what was going on?

### AutoML and AutoKeras

AutoML encapsulates Machine Learning model-making and offers users an interface at the level of tasks. An example of a task is text classification, identifying a text as belonging to a certain class, like being positive or negative. The challenge for an AutoML system is to generate an optimal architecture for a task defined by a user.

AutoKeras is an AutoML system based on the deep learning framework [Keras](https://keras.io/).
 

### Problem statement

AutoKeras delivers a "best model" after a training and evaluation episode, but doesn't provide any accountability for that selection.

There are two reasons for wanting better insight in the search space AutoKeras traverses:
1. Get confidence that AutoKeras is really exploring the search space.
2. Use AutoKeras as an exploratory device and use the results of its searches as input for further experiments.

Existing reporting systems, like [TensorBoard](https://www.tensorflow.org/tensorboard) and [TRAINS](https://github.com/allegroai/trains) are aimed at the individual model level and quickly become unwieldy when dealing with the tens of trials that AutoKeras performs.


### Approach

KerasTuner/AutoKeras leaves a trace of reporting data in JSON format for every trial.

After trial/evaluation we read this into a dataframe from which it can be processed in various ways to gain understanding of the AutoKeras search space and the results of traversing it.

The focus here is on the AutoKeras architecture search space and on model search results. _Not_ on specific tasks, datasets, or quality of the best model.


### Audience / Reading guide

Findings are specific for AutoKeras and will be of interest only to those using AutoKeras and wanting to better understand what it is doing.
The offered environment to get information on trials from AutoKeras will be primarily helpful for those that feel comfortable with python and pandas.

To get straight to the code for reporting on AutoKeras trials, you can skip this text and jump straight to the ["Understanding AutoKeras" notebook](./Understanding%20Autokeras.ipynb).


## AutoKeras architecture search

The origins of the AutoKeras system are described in ["Auto-Keras: An Efficient Neural Architecture Search System](https://arxiv.org/pdf/1806.10282.pdf) by Haifeng Jin, Qingquan Song, and Xia Hu.

The 2018 paper notes that "while there are several AutoML services available on large cloud computing platforms, three things are prohibiting the users from using them":

The cloud services are not free to use, which may not be affordable for everyone who wants to use AutoML techniques.
The cloud-based AutoML usually requires complicated configurations of Docker containers and Kubernetes, which is not easy for people without a rich computer science background.
The AutoML service providers are honest-but-curious, which cannot guarantee the security and privacy of the data. An open-source software, which is easily downloadable and runs locally, would solve these problems and make the AutoML accessible to everyone.
The key idea of the proposed method is to explore the search space via morphing the neural architectures guided by Bayesian optimization algorithm.
The basic assumption behind AutoKeras is that there is a single search space of neural architectures where all can be derived from an initial architecture. Originally, it used Bayesian Optimization to traverse through a tree-shaped search space where one architecture could be morphed into another by specific steps.

What we find in practice, however, is that AutoKeras increasing includes architectures that cannot be morphed into other architectures. Think of BERT for language or ResNet and Xception for images.

In practice, I found that running Autokeras did not result in a convergence to a similar outcome. What is AutoKeras doing? What models are actually trained and with what result?

We would like to better understand:

* The search space
* The search algorithm


### Search space

Originally, AutoKeras, as defined in the paper, was based on a single search space, where one architecture could be transformed into any other through a specific set of transformations. 

During the training performed above, we found three distinct architectures being tried for text classification:

* "Vanilla" - basic model of embedding, convolution, reduction, and a dense layer
* Transformer - transformer based architecture
* BERT - possibly pre-trained version of the BERT architecture

As mentioned earlier, AutoKeras offers a set of pre-defined tasks like Image and Text classification. For each of these tasks, one or more initial architectures are defined in the AutoKeras code in ./autokeras/tuners/task_specific.py

For example for transformers we find in the text classification definition:

```python
{
        "text_block_1/block_type": "transformer",
        "classification_head_1/dropout": 0,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "text_block_1/max_tokens": 20000,
        "text_block_1/text_to_int_sequence_1/output_sequence_length": 200,
        "text_block_1/transformer_1/pretraining": "none",
        "text_block_1/transformer_1/embedding_dim": 32,
        "text_block_1/transformer_1/num_heads": 2,
        "text_block_1/transformer_1/dense_dim": 32,
        "text_block_1/transformer_1/dropout": 0.25,
        "text_block_1/spatial_reduction_1/reduction_type": "global_avg",
        "text_block_1/dense_block_1/num_layers": 1,
        "text_block_1/dense_block_1/use_batchnorm": False,
        "text_block_1/dense_block_1/dropout": 0.5,
        "text_block_1/dense_block_1/units_0": 20,
    },
```

At the top we find a `block_type` which has the value `transformer`, `dropout` for the classification head (the layer which produces the labels), plus the definition of an `optimizer` with a `learning_rate`.

The `text_block_1` prefix refers to the `TextClassifier` in [./autokeras/tasks/text.py](https://github.com/keras-team/autokeras/blob/master/autokeras/tasks/text.py).

The possible parameters can be found in the AutoKeras code. If we look in the definition we find after `text_block_1` a string `transformer_1` and `dense_block_1`. We find the corresponding classes `Transformer` and `DenseBlock` in [./autokeras/blocks/basic.py](https://github.com/keras-team/autokeras/blob/master/autokeras/blocks/basic.py).
The initializer of the `Transformer` class takes as arguments `max_features`, `pretraining`, `embedding_dim`, `num_heads`, `dense_dim`, and `dropout`.

Now that we understand the initial parameters of a search space in AutoKeras it is time to understand their possible values.

After fitting the TextClassifier we can call its tuner to get an overview of the search space. If we skipped training, no summary of the search space is available!

    >>> clf.tuner.search_space_summary()
    Search space summary
    Default search space size: 0
    
In the summary we see the possible values for different parameters. Let's present it in a more readable way. For presenting data from AutoKeras I have created a separate python script containing a number of classes and functions to avoid the notebook from becoming cluttered. Focus of the present article is on the search space of AutoKeras and this code only serves as a utility.


### Search algorithm





