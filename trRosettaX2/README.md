# trRosettaX2 inference code for static structure prediction

Overview
----
[![Python version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)  [![PyTorch version](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)](https://pytorch.org/) 

This folder contains the source code for trRosettaX2, a lightweight end-to-end approach for protein static structure prediction, which achieves competitive performance with AlphaFold2 though it adopts much fewer parameters and computational resources. 

----
Materials
----

trRosettaX2 and trRosettaX2-Dynamics operate in the same environment (`trX2dy`); therefore, it is only necessary to download the pre-trained parameters for trRosettaX2 if the `trX2dy` environment has been created.

```bash
mkdir -p model_pth
cd model_pth
wget http://yanglab.qd.sdu.edu.cn/trRosetta/downloadX2/trX2_orig_models.tar.bz2
tar -xjf trX2_orig_models.tar.bz2
cd ../
````
----

Usage
----
### Running Inference

```bash
# use the `trX2dy` environment
python predict.py -i example/seq.a3m -o example/output
```
The predicted 3D structure will be saved as a PDB file `model_1.pdb` under the `example/output` directory.

For a complete description of all `predict.py` options and arguments, please run:

```bash
python predict.py -h
```


