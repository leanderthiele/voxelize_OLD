# module load hdf5/intel-17.0/1.10.0
icc -Ofast -std=c++17 -qopenmp \
  -no-inline-max-size -no-inline-max-total-size \
  -Wall -Wextra \
  -I/home/lthiele/Overlaps \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64 \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5_hl.a \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5.a \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5_hl_cpp.a \
  -L/usr/local/hdf5/intel-17.0/1.10.0/lib64/libhdf5_cpp.a \
  -lhdf5 -lhdf5_cpp \
  -o voxelize voxelize.cpp
