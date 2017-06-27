# Writing Style
Code for classifying sentence according to their writing style. This is the code used for the style features classification in the following paper:

*The Effect of Different Writing Tasks on Linguistic Style: A Case Study of the ROC Story Cloze Task*
*Roy Schwartz, Maarten Sap, Yannis Konstas, Li Zilles, Yejin Choi and Noah A. Smith*, In proceedings of CoNLL 2017 ([pdf](https://arxiv.org/abs/1702.01841), [bib](http://homes.cs.washington.edu/~roysch/papers/language_constraint/language_constraint.bib))

### Requirements:

-- python2.7, with numpy, sklearn and spacy
-- perl5

### Running:

    1. pre_process.sh <ROC story dev file> <ROC story test file> <work directory = $PWD> <language model scores (dev set)> <language model scores (test set)>
	
-- This script generates files needed for training and testing. The files are stored in the input working directory.
-- The required arguments are the ROC story dev set and train set (see http://cs.rochester.edu/nlp/rocstories/).
-- Running the code without the last two arguments only uses the style classification features described in the paper (length, character n-grams and word n-grams).
-- In order to include the language model features described in the paper, two other arguments should be provided: the language model scores on the dev and test set. In order to generate those, a language model needs to be trained on the ROC story training set, and applied to the ROC story dev and test set. The code for this is found at https://github.com/maarten1709/writing_style_lm. 

    2. run_grid_search.PL <working directory>

-- This script runs grid search on the regularization parameter and returns results for experiment 1 and on the ROC story cloze task on train, dev and test set.

### Misc

-- The pre-processing step generates different train/dev splits each time, so results will vary between runs (and specifically different from the results published in the paper).

### Contact
roysch@cs.washington.edu

