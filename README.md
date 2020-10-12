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
which means you shouldn't move/delete it.


Usage
=====

The python code has a single class, named Box:
```python
from voxelize import Box

box_N   = 1024    # number of grid points
box_L   = 25000.0 # sidelength in the same units as coordinates and radii
box_dim = 1       # set this to the same dimension as the one of your field,
                  #    e.g. 1 if field is a scalar and 3 if field is a vector

b = Box(box_N, box_L, box_dim=box_dim)

# assume we have coordinates, radii, field in memory
#     as numpy arrays (dtype=np.float32!)
# The shapes should be :
#     coordinates = (Nparticles, 3)
#     radii       = (Nparticles, )
#     field       = (Nparticles, box_dim) or (Nparticles, ) if box_dim==1
b.add_particles(coordinates, radii, field)

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
