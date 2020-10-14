import numpy as np
from time import time

from voxelize import Box

# want different seeds when running repeatedly
np.random.seed(abs(hash(time()))%2**32)

# some settings
N_particles = 128*128*128

box_N = 128
box_L = np.random.rand() * float(box_N)

# compute particle size (identical for each particle)
V_particle = box_L**3.0 / float(N_particles)
R_particle = np.cbrt(3.0 * V_particle / 4.0 / np.pi)

for dim in [1, 3, 5] :
    
    # different seed for each dimension
    np.random.seed((abs(hash(time()))+dim)%2**32)

    # define coordinates, radii, field
    c = np.array(np.random.rand(N_particles, 3)*box_L,
                 dtype=np.float32)
    
    # need to ensure correct mean here
    #    use exponential dist since it allows us to generate relatively large
    #    range of possible radii that are positive by construction
    r = np.array(np.cbrt(3.0 / 4.0 / np.pi
                         * np.random.exponential(scale=V_particle,
                                                 size=N_particles)),
                 dtype=np.float32)
    
    # sanity check
    box_vol = box_L**3.0
    part_vol = 4.0*np.pi/3.0 * np.sum(r**3.0)
    print('expect ~ %.2e fractional Delta '\
          '(due to particles not exactly spanning the volume)'%np.fabs(part_vol/box_vol - 1.0))

    f = np.array(np.random.rand(N_particles, dim),
                 dtype=np.float32)

    # create a Box instance
    b = Box(box_N, box_L, box_dim=dim, spherical=False)

    # add the particles
    b.add_particles(c, r, f)

    # check output
    expected = np.mean(f)
    computed = np.mean(b.box)
    diff     = np.fabs(computed/expected - 1.0)
    print('expected = %.4e, computed = %.4e '\
          '--> fractional Delta = %.4e\n'%(expected,
                                           computed,
                                           diff))

print('\t(working in single-precision, so <~ 1e-4 differences are ok)')
