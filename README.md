# tf-sru

## Progress so far

- [x] Setup GCE instance for training
  - [X] Obtain GCP approval for additional GPUs
- [X] Reproduce author's SRU implementation
- [ ] Reproduce classification model
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
