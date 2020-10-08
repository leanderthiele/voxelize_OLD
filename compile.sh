# module load hdf5/intel-17.0/1.10.0

# adapt for your own compilation
export PATH_TO_EIGEN=/home/lthiele/Overlaps

icc -Ofast -std=c++17 -qopenmp \
  -Wall -Wextra \
  -I${PATH_TO_EIGEN} \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64 \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5_hl.a \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5.a \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5_hl_cpp.a \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5_cpp.a \
  -lhdf5 -lhdf5_cpp \
  -o voxelize voxelize.cpp
