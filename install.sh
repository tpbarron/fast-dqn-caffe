#!/bin/sh

rm -rf dependencies/
mkdir -p dependencies/caffe/

# download caffe distribution 
echo "Downloading compatible CAFFE distribution"
wget http://tpbarron.github.io/sources/distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz --directory-prefix=dependencies/caffe
cd dependencies/caffe/
tar xfz distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3.tar.gz
gunzip -r distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3
cd ../../

# download and make minecraft interface
echo "Downloading and building ALE"
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git dependencies/Arcade-Learning-Environment
cd dependencies/Arcade-Learning-Environment/
# apply patch for grayscale images and SDL
echo "Applying grayscale patch"
git apply ../../patches/grayscale.patch
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

echo "Finished install"
echo "Run with ./build/fast_dqn"

