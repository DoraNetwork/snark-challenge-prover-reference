#!/bin/bash
mkdir build
pushd cuda-fixnum
    cmake .
    make
popd
pushd build
  cmake -DMULTICORE=ON -DUSE_PT_COMPRESSION=OFF .. 
  make -j12 main generate_parameters
popd
mv build/libsnark/main .
mv build/libsnark/generate_parameters .
