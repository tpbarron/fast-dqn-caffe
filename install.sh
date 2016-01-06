#!/bin/sh

mkdir -p dependencies/caffe/

# download caffe distribution 
echo "Downloading compatible CAFFE distribution"
wget http://tpbarron.github.io/sources/distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz --directory-prefix=dependencies/caffe
cd dependencies/caffe/
tar xvfz distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz
gunzip -r distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3
cd ../../

# download and make minecraft interface
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
