gpu_gmm
====

Python module to train GMMs using Tensorflow (and therefore a GPU if you want) with an online version of EM.
As for now there is only a version with full matrix for covariances.

Limitation is the memory on your GPU...

Basically I gathered things from people more talented than me and don't have invented anything :

* https://github.com/aakhundov/tf-example-models : Tensorflow implementation of GMM with full covariances matrix
* https://people.csail.mit.edu/danielzoran/NIPSGMM.zip : for online implementation of EM
* https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/mixture/gaussian_mixture.py#L435 : for BIC and AIC score

If you want to do the training and GPU, please make sure that you have install properly : CUDA and Tensorflow-gpu.

## Contents
[Dependencies](#dependencies)
[Installation](#installation)
[Example usage](#example-usage)
[Help](#help)

## Dependencies

* Tensorflow (GPU version if you want to use it with GPU...)
* Numpy
* Sklearn

## Installation

Clone repo
```bash
cd ${INSTALL_PATH}
git clone https://github.com/ludovicdmt/gpu_gmm.git
```

Install gpu_gmm:
```bash
cd ${INSTALL_PATH}/ggmm
sudo pip install .
```
OR
```bash
cd ${INSTALL_PATH}/ggmm
sudo python setup.py install
```

## Example Usage
See test.py

## Help

If you experience issues during installation and/or use of gpu_gmm, you can post a new issue on the [gpu_gmm GitHub issues webpage](https://github.com/ludovicdmt/gpu_gmm/issues). I will reply to you as soon as possible and I'm very interested in to improve this tool.  