# voxelize

Compile:
You'll need the Eigen header files (http://eigen.tuxfamily.org/),
and you'll need to adapt the script compile.sh to reflect the location
of these header files.
```shell
module load intel/18.0/64/18.0.3.222
module load hdf5/intel-17.0/1.10.0
sh compile.sh
```

Use:
```shell
module load intel/18.0/64/18.0.3.222
module load intel-mpi/intel/2018.3/64
module load hdf5/intel-17.0/1.10.0
export OMP_NUM_THREADS=<...>
./voxelize PTYPE \
           INPUT_PREFIX \
           OUTPUT_PREFIX \
           NCHUNKS \
           BOX_SIZE \
           NSIDE \
           NSUBBOXES \
           [AXIS]
```
with the following command line-arguments
* PTYPE: (int)
    * 0 = electron pressure
    * 1 = dark matter density
    * 2 = electron number density
    * 3 = dark matter momentum density
    * 4 = electron momentum density
    * 5 = dark matter velocity
* INPUT_PREFIX: (str) It is assumed that the input simulation files have the filenames
                      INPUT_PREFIX\<n\>.hdf5, where n is an index ranging from 0 to NCHUNKS-1
* OUTPUT_PREFIX: (str) The output files will be written in the format
                       OUTPUT_PREFIX\_\<i\>\_\<j\>\_\<k\> where i,j,k run from 0 to NSUBBOXES-1.
                       These files are binary files in row-major (C) order, covering a cubical
                       sub-box of the original simulation box.
                       Note that the data is single precision, so if reading from numpy you need to
                       call np.fromfile(\<fname\>, dtype=np.float32)
* NCHUNKS: (int) number of simulation chunks (see INPUT_PREFIX)
* BOX_SIZE: (float) sidelength of the simulation box, in the same units the particle coordinates are given
                    in (I think it's kpc/h for Illustris).
* NSIDE: (int) discretization of the output
* NSUBBOXES: (int) if set to 1, only a single output file is produced.
                   If working with large boxes or at high resolution, it may be convenient to split the
                   output into smaller files.
                   This can be accomplished by setting NSUBBOXES to larger than 1.
                   In that case, NSUBBOXES^3 output files will be produced,
                   each corresponding to a cube of sidelength NSIDE/NSUBBOXES
                   (note that NSUBBOXES should divide NSIDE without remainder).
* [AXIS]: (int) if PTYPE corresponds to a vectorial quantity, e.g. DM velocity,
                this argument specifies the component of the vector which is used.
                Otherwise it is ignored, and need not be set.

Update
======
There's a new version now for you convenience, which is more flexible.
Use:
```shell
./voxelize_new PTYPE \
               INPUT_PREFIX \
               OUTPUT_PREFIX \
               NCHUNKS \
               BOX_SIZE \
               NSIDE \
               NSUBBOXES \
               OPERATION
```
Compared to the above, the following arguments are different:
* PTYPE: (int)
    now it's really the particle type:
    * 0 = gas
    * 1 = dark matter
* OPERATION: (str) For extra convenience, you can now directly specify the combination of fields you want.
                   OPERATION is a string that consists of factors, delimited by the character \*.
                   No spaces allowed.
                   The factors, again, split into two parts: the base and the exponent,
                   delimited by the character ^.
                   The exponent, together with the ^ can be omitted.
                   The base can be of two types:
    * something that can be interpreted as a float (e.g. 3.14)
    * something that can be interpreted as a field name (e.g. Density)
                   For example, you can take OPERATION as
                   ```
                   1.23^1.5*Density^-1*Whateverotherfieldnameyouwouldwant^3.14
		   ```
                   One final catch: For vectorial quantities, e.g. Velocity, you should follow their name
                   by [n], where n (0..2) gives the direction you want to extract, e.g.
                   ```
                   Velocity[1]^3.0
                   ```
