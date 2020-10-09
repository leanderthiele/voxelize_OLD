# module load intel/19.1/64/19.1.1.217
# module load hdf5/intel-17.0/1.10.0

# adapt for your own compilation
export PATH_TO_EIGEN=/home/lthiele/Overlaps

icc -Ofast -std=c++17 -qopenmp \
  -Wall -Wextra \
  -I${PATH_TO_EIGEN} \
  -lhdf5 -lhdf5_cpp \
  -o voxelize voxelize.cpp
