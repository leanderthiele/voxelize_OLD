**UPDATE**: a new, better version of this code is available at
[this repository](https://github.com/leanderthiele/voxelize).


Installation
============

Simply run
```shell
sh compile.sh
```
This will compile the C++ code, and run pip.
You may need to edit the shell script if the Eigen header files
are not in one of the standard locations.

In order to test that everything works well,
you can run
```shell
python test.py
```

Note: the python package *hardcodes* the location of libvoxelize.so,
which means you probably shouldn't move/delete it.
You can also move libvoxelize.so, provided you append the new location
to LD\_LIBRARY\_PATH.

**Installation on rusty**

1) Place Eigen3 headers in software. Version 3.3.8 seems to have problems. 3.3.7 works fine. Change name from eigen-3.3.7 to eigen3

2) Add this to the compile.sh script: -I/mnt/home/fvillaescusa/software

3) Change < > by " " in the file overlap_lft.hpp (this is already done by default now)

4) sh compile.sh


Usage
=====

The python code has a single class, named Box:
```python
from voxelize import Box

box_N   = 1024    # number of grid points per side
box_L   = 25000.0 # sidelength in the same units as coordinates and radii
box_dim = 1       # set this to the same dimension as the one of your field,
                  #    e.g. 1 if field is a scalar and 3 if field is a vector
spherical = True  # by default, represents the particles as uniform spheres.
                  #    Setting this option to False uses a representation as cubes,
                  #    which is much faster but perhaps slightly less accurate
                  #    (this needs to be tested)

b = Box(box_N, box_L, box_dim=box_dim, spherical=spherical)

# assume we have coordinates, radii, field in memory
#     as numpy arrays (dtype=np.float32!)
# The shapes should be :
#     coordinates = (Nparticles, 3)
#     radii       = (Nparticles, )
#     field       = (Nparticles, box_dim) or (Nparticles, ) if box_dim==1
b.add_particles(coordinates, radii, field)

# add_particles() can be called repeatedly,
#    in case you want to chunk the inputs to save memory

# At any point, the filled box can be retreived as
b.box
# It has shape (box_N, box_N, box_N, box_dim) if box_dim != 1,
#         else (box_N, box_N, box_N)
```

Notes:
* The code will fail if the data types are not single precision (np.float32)
* There could be silent errors if the input arrays are not contiguous in memory
  (i.e., be careful with things like transposing, and non-trivial slicing)
* You should set the environment variable OMP\_NUM\_THREADS to control multi-threading
* The scheduling of the OpenMP threads can be controlled using the environment variable
  OMP\_SCHEDULE.
  I would recommend the following:
  * set to "static" if the particles are more or less randomly ordered.
    Then each thread should receive approximately equal load and "static" scheduling
    incurs the least overhead.
  * set to "dynamic" if the ordering of the particles is very non-random.
    e.g., it can be be that there's a dense halo with a lot of very small particles
    at the beginning of the array. These particles will require less work than those in
    underdense regions (where their size is fairly large, so many overlaps need to be computed).
    In that case "dynamic" scheduling could be more efficient.

  I would recommend to start with "static" scheduling and only adjust if you see that some
  threads receive much less work than others.
