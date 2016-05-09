fast-dqn-caffe
==
An optimized C++ Deep Q-Network (DQN) implementation for Caffe.

The goal is to provide a fast version of DQN that works with relatively little setup.  Where possible, reproducing Deepmind's results is desirable, especially for making comparisons. Still the priority is on speed, since speed is a substantial barrier to the use of reenforcement learning for practical applications.

This is modified from [watts4speed/fast-dqn-caffe](https://github.com/watts4speed/fast-dqn-caffe) to interface with a minecraft implementation.

##Requirements


- Caffe, **commit: ff16f6e43dd718921e5203f640dd57c68f01cdb3**
- minecraft_dqn_interface [tpbarron/minecraft_dqn_interface](https://github.com/tpbarron/minecraft_dqn_interface)
- Ubuntu is the only version tested so far

The script `install.sh` should fetch the `minecraft_dqn_interface` from github and put it in a dependencies directory.

Note:
**fast_dqn** doesn't work with the current head of caffe/master since something changed since September 2015 that blows up the net during training.  Let me know if you find the problem, or if the current code works for you.

To date **fast_dqn** has only been tested on ubuntu 14.04 and 15.10.

The GPU is turned on and has been tested with cudnn v3 and CU toolkit v7.5.


##To build
```
mkdir build
cd build
cmake ..
make
cd ..
```

##To run
```
./build/fast_dqn
```

- In the scripts folder there is scripts/plot.py.  Run this script to watch the training progress.

```
./scripts/plot.py
```


The plot will update every 15 seconds or so, showing a moving average of the game scores.  It also writes ./tran_progress.png which also contains the same plot.  If you leave the plotting program running across multiple training sessions the plot will contain previous runs along with the latest run.  This is helpful when trying stuff out.

![plot of training](tran_progress-example.png)

In the plot above, three different training runs for pong.bin are shown.  The lines represent a running average of the episode scores.  The blue line is the current training run.  It takes about 15 minutes on a Titan-X with an X99 CPU for results to show up with the current code near episode 30.

## Running trained models

The models are snapshoted into the subdirectory ./model.  To run a trained model that was snapshotted after 1000000 steps for pong for example, use the following command:

```
./build/_fast_dqn -evaluate -model ~/model/dqn_iter_1000000.caffemodel
```

##Training details

During training the loss is clipped to 10. After a few seconds of running you should see few messages about the loss being clipped.  My experience is that if continuous clipping of the the loss occurs its diagnostic of a training process not working due to bugs or some other reason.

If your plot for pong looks similar to the above then you're probably in good shape.

My suggestion is to test changes to the code against pong to make sure you haven't broken anything.  It's really easy to do and I've wasted way too much time having done so due to simple bugs.

My approach is to take careful steps and verify new stuff against the pong baseline first.

###Example training log
```
I1214 15:18:15.946360 25637 fast_dqn_main.cpp:174] training score(106): -21
I1214 15:18:23.908684 25637 fast_dqn_main.cpp:174] training score(107): -20
I1214 15:18:23.908716 25637 fast_dqn_main.cpp:187] epoch(21:105472): average score -20.4401 in 0.212203 hour(s)
I1214 15:18:23.908723 25637 fast_dqn_main.cpp:195] Estimated Time for 1 million iterations: 2.01193 hours
I1214 15:18:30.543287 25637 fast_dqn_main.cpp:174] training score(108): -20
I1214 15:18:37.474827 25637 fast_dqn_main.cpp:174] training score(109): -21
```
