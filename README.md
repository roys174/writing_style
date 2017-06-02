# writing_style
Code for classifying sentence according to their writing style.

This is the code used for the style features classification in the following paper:: The Effect of Different Writing Tasks on Linguistic Style: A Case Study of the ROC Story Cloze Task (Roy Schwartz, Maarten Sap, Yannis Konstas, Leila Zilles, Yejin Choi and Noah A. Smith, arXiv 2017, https://arxiv.org/abs/1702.01841)


### Requirements:

- python2.7, with numpy, sklearn and spacy
- perl5

### Running:
1. pre_process.sh <roc story dev file> <roc story test file> <work directory = $PWD>
	
This script generates files needed for training and testing inside a given working directory.

2. run_grid_search.PL <working directory>

This script runs grid search on the regularization parameter and returns results for experiment 1 and on the ROC story cloze task on train, dev and test set.

### Misc

- The pre-processing step generates different train/dev splits each time, so results will vary between runs (and specifically between runs and the results published in the paper)
- This code does not generate the language model part described in the paper, just the style features.

### Contact
roysch@cs.washington.edu

