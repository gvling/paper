# docker
image: tensorflow/tensorflow:1.12.0-gpu-py3<br>
tfboard: tensorboard --logdir ./paper/logs/<br>
```
$ nvidia-docker run -itd --name paper -e PASSWORD=your-password -e http_prox=http-proxy -e https_proxy=https-proxy -v path-to-pj:/notebooks -p host-port:8888 -p host-port:6006 tensorflow/tensorflow:1.12.0-gpu-py3
```


# jupyterExtensions
pip install jupyter_contrib_nbextensions<br>
jupyter contrib nbextension install --user<br>
jupyter nbextension enable codefolding/main<br>
jupyter contrib nbextensions migrate<br>


# paper
[![reference](https://img.shields.io/badge/reference-arXiv-green.svg?style=flat&logo=pinboard)](http://arxiv-sanity.com/top)
[![reference](https://img.shields.io/badge/reference-arXivTimes-green.svg?style=flat&logo=github)](https://github.com/arXivTimes/arXivTimes)
