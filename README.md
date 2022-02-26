# deep-learning-covid-19
This repo contains the Python scripts to perform my thesis study '*Detection of COVID-19 from Chest X-rays using Deep Learning*' (HCMIU VNU-HCMC, Feb - Oct 2021).  

Some heads up:
- List of interested performance metrics: loss, accuracy, precision, recall, F1 score, AU-ROC, AU-PRC
- I also practiced some OOP in my code. The classes and methods are called in a Colab notebook. (It did saved time in the long run.)  
- Full list of references can be found in my thesis

What each script does:
- **logconf.py**:
    - Configurate Python logging module for the whole program (fileHandler to .txt file, streamHandler to standard output)
- **dsets.py**:
    - Define and import dataset, split to train-validation-test sets
    - Augment data
    - Find RGB-normalized parameters (once)
    - Unit test "get" and "forward"
- **model.py**:
    - Load pretrained neural networks from local <model>.pt files and modify their last layers
- **finetune.py**:
    - Perform hyperparameters tuning (is Bayesian optimization of multiple runs of stratified cross validation, each run with a different set of hyperparameters)
    - Calculate tuning metrics (only training set)
    - If interrupted unexpectedly, progress could be restored from periodic checkpoints. Timer included.
    - Output is a list of hyperparameter configuration and corresponding metrics (to local savefile), where the best training conditions can be chosen from
- **training.py**:
    - Perform model training, with SGD optimizer, CyclicLR scheduler, and early stopping 
    - Calculate training metrics (of training and validation set)
    - If interrupted unexpectedly, progress could be restored from periodic checkpoints. Timer included.
    - Output is the training progress and metrics, and a model candidate (all to local savefiles)
- **evaluation.py**:
    - Feedforward the fittest candidate from each of model through the test set
    - Calculate & save metrics
- **visualize.py** plots the following:
    - (GIFs, 1 epoch/frame) Confusion matrices of training and validation sets
    - Performance metrics by epoch
    - (GIFs, 1 epoch/frame) ROC curves and AU-ROC
    - (GIFs, 1 epoch/frame) PR curves and AU-PRC
    - Stopping conditions for early stopping by epoch