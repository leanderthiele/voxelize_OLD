g++ -O4 -ffast-math -std=c++17 -fopenmp \
  -Wall -Wextra \
  -I/usr/include/hdf5/serial \
  -o voxelize_1 voxelize_1.cpp \
  -lhdf5_serial -lhdf5_cpp
