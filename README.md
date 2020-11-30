# Discovering AutoKeras Search Space

## AutoML and AutoKeras
AutoML encapsulates Machine Learning model-making and offers users an interface at the level of tasks. Text classification, identifying a text as belonging to a certain class, like being positive or negative, is one example of a task. The challenge for an AutoML system is to generate an optimal architecture for a task defined by a user.

AutoKeras is an AutoML system based on the deep learning framework Keras.
 

## Problem statement

AutoKeras delivers a "best model" after a training and evaluation episode, but doesn't provide any accountability for that selection.

There are two reasons for wanting better insight in the search space AutoKeras traverses:
1. Get confidence that AutoKeras is really exploring the search space.
2. Use AutoKeras as an exploratory device and use the results of its searches as input a next cycle of exploration.

Existing reporting systems, like TensorFlow and TRAINS are aimed at the individual model level and quickly become unwieldy when dealing with tens of trials.

## Approach

KerasTuner/AutoKeras leaves a trace of reporting data in JSON format on every trial.

After trial/evaluation we read this into a dataframe from which it can be processed in various ways. 
