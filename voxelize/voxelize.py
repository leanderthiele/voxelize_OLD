import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes as ct

from os.path import join
from sys import stdout
from pkg_resources import resource_string

class Box(object) :
    # locate the shared object
    __PATHTOSO = resource_string('voxelize', 'PATHTOSO.txt')
    __PATHTOSO = __PATHTOSO.rstrip()
    if isinstance(__PATHTOSO, bytes) :
        __PATHTOSO = str(__PATHTOSO, encoding=stdout.encoding)
    __lib = ct.CDLL(join(__PATHTOSO, 'libvoxelize.so'))

    # read the mangled name
    __MANGLEDNAME = resource_string('voxelize', 'MANGLEDNAME.txt')
    __MANGLEDNAME = __MANGLEDNAME.rstrip()
    if isinstance(__MANGLEDNAME, bytes) :
        __MANGLEDNAME = str(__MANGLEDNAME, encoding=stdout.encoding)
    
    __voxelize = eval('lib.%s'%__MANGLEDNAME, {'lib': __lib})
    __voxelize.restype = ct.c_int
    __voxelize.argtypes = [ct.c_long, ct.c_long, ct.c_float, ct.c_long,
                           ndpointer(ct.c_float, flags='C_CONTIGUOUS'),
                           ndpointer(ct.c_float, flags='C_CONTIGUOUS'),
                           ndpointer(ct.c_float, flags='C_CONTIGUOUS'),
                           ndpointer(ct.c_float, flags='C_CONTIGUOUS'),]

    def __init__(self,
                 box_N,
                 box_L,
                 box_dim=1) :
        self.box_N = box_N
        self.box_L = box_L
        self.box_dim = box_dim

        if box_dim == 1 :
            self.box = np.zeros((box_N, box_N, box_N),
                                dtype=np.float32, order='C')
        else :
            self.box = np.zeros((box_N, box_N, box_N, box_dim),
                                dtype=np.float32, order='C')

    def add_particles(self,
                      coordinates,
                      radii,
                      field) :
        
        N_particles = coordinates.shape[0]

        # sanity checks
        assert len(coordinates.shape) == 2
        assert coordinates.shape[1] == 3
        assert len(radii.shape) == 1
        assert N_particles == radii.shape[0]
        assert N_particles == field.shape[0]
        if self.box_dim != 1 :
            assert len(field.shape) == 2
            assert field.shape[1] == self.box_dim
        else :
            if len(field.shape) != 1 :
                assert len(field.shape) == 2
                assert field.shape[1] == 1

        # call the compiled function
        err = Box.__voxelize(N_particles, self.box_N, self.box_L, self.box_dim,
                             coordinates, radii, field, self.box)
        if err :
            raise RuntimeError('voxelize returned with non-zero exit code.')
