IPMI 2017
=========

Code for reproducing the key results of our IPMI 2017 paper on structured sparsity penalization for large functional neuroimaging datasets.

Bzdok D, Eickenberg M, Varoquaux G, Thirion B.
**Hierarchical Region-Network Sparsity for High-Dimensional in Brain Imaging**
Information Processing in Medical Imaging (**NIPS 2017**).
[Paper on ResearchGate](https://www.researchgate.net/publication/313476901_Hierarchical_Region-Network_Sparsity_for_High-Dimensional_Inference_in_Brain_Imaging)

Please cite this paper when using this code for your research.

To follow established conventions of scikit-learn estimators, the StructuredEstimator class exposes the functions fit(), predict(), and score().
This should allow for seamless integration into other scikit-learn-enabled machine-learning pipelines.

For questions and bug reports, please send me an e-mail at _danilobzdok[at]gmail.com_.

## Prerequisites

1. Make sure that recent versions of the following packages are available:
	- Python (version 2.7 or higher)
	- Numpy (e.g. `pip install numpy`)
	- Theano (e.g. `pip install Theano`)
	- Nibabel (e.g., `pip install nibabel`)
  - Nilearn (e.g., `pip install nilearn`)
  - Scikit-learn (e.g., `pip install scikit-learn`)
  - SPAMS (see http://spams-devel.gforge.inria.fr/downloads.html)

2. Set `floatX = float32` in the `[global]` section of Theano config (usually `~/.theanorc`). Alternatively you could prepend `THEANO_FLAGS=floatX=float32 ` to the python commands. 

3. Clone this repository, e.g.:
```sh
git clone https://github.com/banilo/ipmi2017.git
```




