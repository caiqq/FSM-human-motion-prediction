

## FSM-human-motion-prediction

This is the code for the paper

Xiurui Xie, Guisong Liu, Qing Cai, Hong Qu, Malu Zhang
An End-to-end Functional Spiking Model for Sequential
Feature Learning.


### Status
The code is on tidying up

### Dependencies

* [h5py](https://github.com/h5py/h5py) -- to save samples
* [Tensorflow](https://github.com/tensorflow/tensorflow/) 1.2 or later.

### Get this code and the data

First things first, clone this repo and get the human3.6m dataset on exponential map format.

```bash
git clone https://github.com/caiqq/FSM-human-motion-prediction.git
cd human-motion-prediction
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

### Run the script

For a quick demo, you can train for a few iterations and visualize the outputs
of your model. Default iteration is 6000.

To train, run
```bash
python src/translate.py --iterations 2000
```



### Reference

The pre-processed human 3.6m dataset and some of our evaluation code was ported/adapted from [1].

[1] J. Martinez, M. J. Black, J. Romero, On human motion prediction using recurrent
neural networks, in: CVPR, 2017.
