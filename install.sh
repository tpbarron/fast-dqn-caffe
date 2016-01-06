#!/bin/sh

rm -rf dependencies/
mkdir -p dependencies/caffe/

# download caffe distribution
echo "Installing dependencies for Caffe"
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
 
echo "Downloading compatible CAFFE distribution"
wget http://tpbarron.github.io/sources/distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz --directory-prefix=dependencies/caffe
cd dependencies/caffe/
tar xvfz distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz
gunzip -r distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3
cd ../../

# download and make minecraft interface
echo "Installing dependencies for minecraft interface"
sudo apt-get install python-dev

echo "Downloading and building minecraft interface"
git clone https://github.com/tpbarron/minecraft_dqn_interface.git dependencies/minecraft_dqn_interface
cd dependencies/minecraft_dqn_interface/
mkdir build && cd build
cmake .. && make
cd ../../../

# make fast-dqn 
echo "Building dqn"
mkdir build && cd build
cmake .. && make
cd ../

echo "Finished install"
echo "Run with ./build/fast_dqn
