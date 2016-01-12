#!/bin/sh

rm -rf dependencies/
mkdir -p dependencies/caffe/

# download caffe distribution 
echo "Installing dependencies for Caffe"
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

echo "Downloading compatible Caffe distribution"
# gpu capable version
wget http://tpbarron.github.io/sources/distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz --directory-prefix=dependencies/caffe
cd dependencies/caffe/
tar zxf distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz
cd ../../
# caffe gpu distribution with recurrent 
wget http://tpbarron.github.io/sources/distribute_recurrent_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz --directory-prefix=dependencies/caffe
cd dependencies/caffe/
tar zxf distribute_recurrent_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz
cd ../../
# cpu capable version
wget http://tpbarron.github.io/sources/distribute_cpu_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz --directory-prefix=dependencies/caffe
cd dependencies/caffe/
tar zxf distribute_cpu_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz
cd ../../

# download and make minecraft interface
echo "Installing dependencing for ALE"
sudo apt-get install libsdl1.2-dev
echo "Downloading and building ALE"
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git dependencies/Arcade-Learning-Environment
cd dependencies/Arcade-Learning-Environment/
# apply patch for grayscale images and SDL
#echo "Applying grayscale patch"
#git apply ../../patches/grayscale.patch
echo "Applying SDL patch for build"
git apply ../../patches/sdl.patch
echo "Building ALE"
mkdir build && cd build
cmake .. && make
cd ../../../

# make fast-dqn 
echo "Building DQN"
# create build dir if not exists and clean
mkdir -p build && cd build && rm -rf *
cmake .. && make
cd ../

# for the plotting script
echo "Installing dependencies for plotting script"
sudo apt-get install python-dateutil
sudo apt-get install python-pyparsing
sudo pip install matplotlib==1.5
sudo apt-get install python-numpy

echo "Finished install"
echo "Run with ./build/fast_dqn"

