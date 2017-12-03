# tf-sru

## Progress so far

- [x] Setup GCE instance for training
  - [X] Obtain GCP approval for additional GPUs
- [X] Reproduce author's SRU implementation
- [X] Reproduce classification model
- [ ] Reproduce question answering model
- [X] Reproduce langauge model
- [ ] Reproduce translation model
- [ ] Reproduce speech model
- [ ] Rewrite in TensorFlow


## Steps taken to reproduce language model:

*Reproduced on Intel x86_64 with NVIDIA K80 GPU running Ubuntu 17.04.*

1. Install CUDA 8.0 following [NVIDIA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
2. Install Anaconda for Python 2.7: `conda create -n py27 python=2.7 anaconda`.
3. Activate new conda environment: `source activate py27`
4. Clone source repo: `git clone https://github.com/taolei87/sru`
5. Install requirements: `pip install -r sru/requirements.txt`.
6. Download LSTM training data: `git clone https://github.com/yoonkim/lstm-char-cnn`
7. Export required paths: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && export PYTHONPATH=./sru`
8. Run LM training example: `python sru/language_model/train_lm.py --train lstm-char-cnn/data/ptb/train.txt --test lstm-char-cnn/data/ptb/test.txt --dev lstm-char-cnn/data/ptb/valid.txt`

## Steps taken to reproduce classification:

1. Install CUDA 8.0 following [NVIDIA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
2. Install Anaconda for Python 2.7: `conda create -n py27 python=2.7 anaconda`.
3. Activate new conda environment: `source activate py27`
4. Clone source repo: `git clone https://github.com/taolei87/sru`
5. Install requirements: `pip install -r sru/requirements.txt`.
8. Install PyTorch: `conda install pytorch torchvision cuda80 -c soumith`
6. Download the dataset from  https://github.com/harvardnlp/sent-conv-torch/tree/master/data
7. Download a pre-trained word embedding such as word2vec from https://github.com/mmihaltz/word2vec-GoogleNews-vectors or lexVec from https://github.com/alexandres/lexvec (in text format, not binary)
8. Export required paths: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && export PYTHONPATH=./sru`
9. Run the classification example: python train_classifier2.py --path PATH_TO_DATASET 
--embedding PATH_TO_WORD_EMBEDDING --max_epoch 10 --cv 0

