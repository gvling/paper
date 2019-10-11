# docker
image: tensorflow/tensorflow:1.12.0-gpu-py3
tfboard: tensorboard --logdir ./paper/logs/

# jupyterExtensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable codefolding/main
jupyter contrib nbextensions migrate


# paper
[![reference](https://img.shields.io/badge/reference-arXiv-green.svg?style=flat&logo=pinboard)](http://arxiv-sanity.com/top)
[![reference](https://img.shields.io/badge/reference-arXivTimes-green.svg?style=flat&logo=github)](https://github.com/arXivTimes/arXivTimes)
