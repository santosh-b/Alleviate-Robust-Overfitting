# Fast Adversarial Training + RB
This is a supplemental material containing the code to run Fast is better than
free: revisiting adversarial training, submitted to ICLR 2020.

The framework used is a modified version of the [Free Adversarial Training](https://github.com/mahyarnajibi/FreeAdversarialTraining/blob/master/main_free.py) repository, which in turn was adapted from the [official PyTorch repository](https://github.com/pytorch/examples/blob/master/imagenet).

## Installation
1. Install [PyTorch](https://github.com/pytorch/examples/blob/master/imagenet).
2. Install the required python packages. All packages can be installed by running the following command:
```bash
pip install -r requirements.txt
```
3. Download and prepare the ImageNet dataset. You can use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh), 
provided by the PyTorch repository, to move the validation subset to the labeled subfolders.
4. Prepare resized versions of the ImageNet dataset, you can use `resize.py` provided in this repository. 
5. Install [Apex](https://github.com/NVIDIA/apex) to use half precision speedup. 

## Prepare
Download pretrained Early Bird Ticket from [Google Drive](https://drive.google.com/drive/folders/1aTl47KR8oaHkOxqKOUDgQyI96dW7JgHJ). 
Please refer [EB-Tickets](https://github.com/RICE-EIC/Early-Bird-Tickets) for more details.  
Put the downloaded models in the current `ImageNet` folder. It will look like 
~~~
ImageNet/
----- ResNet50/
---------- pruned_3010_0.3/
--------------- pruned.pth.tar
---------- pruned_5008_0.5/
--------------- pruned.pth.tar
---------- pruned_7008_0.7/
--------------- pruned.pth.tar

~~~

## Training a model

##### Dense Model Adversarial Training 
Scripts to robustly train an ImageNet classifier for epsilon radii of 2/255 and 4/255 are provided in `run_fast_2px.sh` and `run_fast_4px.sh`. 
These run the main code module `main_fast.py` using the configurations provided in the `configs/` folder. 
To run the 50 step PGD adversary with 10 restarts, we also provide `run_eval.sh`. 
All parameters can be modified by adjusting the configuration files in the `configs/` folder. 

##### Robust Ticket Adversarial Training
You may need to revise the `DATA160, DATA320, DATA` in these scripts to revise the path of the ImageNet dataset.   
All parameters can be modified by adjusting the configuration files in the `configs_rticket/` folder. 

1. 30% Sparsity
~~~
run_rticket_fast_2px_eb30.sh
~~~

2. 50% Sparsity
~~~
run_rticket_fast_2px_eb50.sh
~~~

3. 70% Sparsity
~~~
run_rticket_fast_2px_eb70.sh
~~~


## Model weights
We also provide the model weights after training with these scripts, which can be found in this [Google drive folder](https://drive.google.com/open?id=1W2zGHyxTPgHhWln1kpHK5h-HY9kwfKfl). To use these with the provided evaluation script, either adjust the path to the model weights in the `run_eval.sh` script or rename the provided model weights accordingly. 
