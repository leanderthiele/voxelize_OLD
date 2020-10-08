# voxelize

Compile:
```shell
module load hdf5/intel-17.0/1.10.0
sh compile.sh
```

Use:
```shell
module load hdf5/intel-17.0/1.10.0
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
                       OUTPUT_PREFIX_<i>_<j>_<k> where i,j,k run from 0 to NSUBBOXES-1.
                       These files are binary files in row-major (C) order, covering a cubical
                       sub-box of the original simulation box.
                       Note that the data is single precision, so if reading from numpy you need to
                       call np.fromfile(<fname>, dtype=np.float32)
* NCHUNKS: (int) number of simulation chunks (see INPUT_PREFIX)
* BOX_SIZE: (float) sidelength of the simulation box, in the same units the particle coordinates are given
                    in (I think it's kpc/h for Illustris).
* NSIDE: (int) discretization of the output
* NSUBBOXES: (int) if set to 1, only a single output file is produced.
                   If working with large boxes or at high resolution, it may be convenient to split the
                   output into smaller files.
                   This can be accomplished by setting NSUBBOXES to non-zero.
                   In that case, NSUBBOXES^3 output files will be produced,
                   each corresponding to a cube of sidelength NSIDE/NSUBBOXES
                   (note that NSUBBOXES should divide NSIDE without remainder).
* [AXIS]: (int) if PTYPE corresponds to a vectorial quantity, e.g. DM velocity,
                this argument specifies the component of the vector which is used.
                Otherwise it is ignored, and need not be set.
