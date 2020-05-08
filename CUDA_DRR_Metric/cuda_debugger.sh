#!/bin/bash
source /home/leko/anaconda3/bin/activate python37
cd SiddonClassLib/src/
safe-rm -r build
mkdir build
cd build
cmake ..
make
cp libSiddonGpu.a ../../../SiddonPythonModule/lib/
cd ../SiddonLib
cp siddon_class.cuh ../../../SiddonPythonModule/include
cd ../../../SiddonPythonModule
safe-rm -r build
safe-rm SiddonGpuPy.cpp
safe-rm SiddonGpuPy.cpython-37m-x86_64-linux-gnu.so
python setup.py build_ext --inplace
cp SiddonGpuPy.cpython-37m-x86_64-linux-gnu.so ../../3D_2D_Registration/wrapped_module
echo "done!"
