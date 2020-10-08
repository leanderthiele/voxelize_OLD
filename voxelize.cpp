#include <iostream>
#include <stdio.h>
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::exception;

#include <exception>
#include <string>
#include <fstream>
#include <cmath>
#include <math.h>
#include <omp.h>
#include "H5Cpp.h"
#include "overlap_lft.hpp"

// Physical constants
#define KBOLTZM (1.3806485e-16)        // cgs
#define GNEWTON (4.3e4f)               // (kpc/1e10 Msun) (km/s)^2
#define XH      (0.76)                 // hydrogen mass fraction, dimensionless
#define GAMMA   (5.0/3.0)              // adiabatic index, dimensionless
#define MSUN    (1.98847e43*0.6774)    // cgs (1e10 Msun/h)
#define MPROTON (1836.15267343)        // in electron masses

// particle types
#define GAS    0 // electron pressure
#define DM     1 // DM density
#define NE     2 // electron number density
#define MOMDM  3 // dark matter momentum
#define MOMGAS 4 // electron momentum density
#define VELDM  5 // dark matter velocity

using namespace H5;

struct SimChunk
// This is a single chunk from a snapshot.
{//{{{
    SimChunk(const string name)
    {
        this->chunk_name = name;
        this->file = new H5File(chunk_name, H5F_ACC_RDONLY);
        this->header = new Group(this->file->openGroup("/Header"));
    }
    ~SimChunk()
    {
        delete file;
        delete header;
    }
    template <typename T>
    void get_header_field(const string name, T *out) const
    {
        Attribute attr = this->header->openAttribute(name);
        attr.read(attr.getDataType(), out);
    }
    string chunk_name;
    H5File *file;
    Group  *header;
};//}}}

struct Field
// This is a field (e.g. Coordinates, Density, ... for Particle Type ptype)
// If ptype is not given, assumes the field is a group field (e.g. Halo Mass, ...)
{//{{{
    Field(const SimChunk *s, const int ptype, const string name)
    {
        this->s = s;
        if (ptype==GAS || ptype==NE)
        {
           this->ptype = 0;
        }
        else if (ptype==DM)
        {
            this->ptype = 1;
        }
        else if (ptype==MOMDM)
        {
            this->ptype = 1;
        }
        else if (ptype==VELDM)
        {
            this->ptype = 1;
        }
        else if (ptype==MOMGAS)
        {
            this->ptype = 0;
        }
        else
        {
            cout << "Unknown ptype in Field." << endl;
        }
        this->name = name;
        this->dataset =  new DataSet(s->file->openDataSet("PartType" + to_string(this->ptype) + "/" + this->name));
        this->dataspace = new DataSpace(this->dataset->getSpace());
        this->Ndims = this->dataspace->getSimpleExtentDims(this->dim_lengths, NULL);
        this->occupies_memory = 0;
    }
    Field(const SimChunk *s, const string name)
    {
        this->s = s;
        this->ptype = -1;
        this->name = name;
        this->dataset = new DataSet(s->file->openDataSet(this->name));
        this->dataspace = new DataSpace(this->dataset->getSpace());
        this->Ndims = this->dataspace->getSimpleExtentDims(this->dim_lengths, NULL);
        this->occupies_memory = 0;
    }
    ~Field()
    {
        delete this->dataset;
        delete this->dataspace;
        if (this->occupies_memory)
        {
            delete   this->memspace;
            delete[] this->values;
        }
    }
    void read_to_memory(void)
    // reads the complete field to memory
    {
        this->memspace = new DataSpace(this->Ndims, this->dim_lengths, NULL);
        long int Nvalues = 1;
        for (int ii=0; ii<Ndims; ii++) { Nvalues *= this->dim_lengths[ii]; }
        try
        {
            this->values = new float[Nvalues];
            this->occupies_memory = 1;
        }
        catch (exception& e)
        {
            cout << "Standard exception :" << e.what() << endl;
        }
        this->dataset->read(this->values, PredType::NATIVE_FLOAT, *this->memspace, *this->dataspace);
    }
    float &operator[] (const long int index) const
    // overload the array index operator for convenience
    {
        return this->values[index];
    }
    const SimChunk *s;
    DataSet *dataset;
    DataSpace *dataspace;
    DataSpace *memspace;
    int ptype;
    string name;
    int Ndims;
    hsize_t dim_lengths[10]; // there is no dataset with more than 10 dims
    float *values;
    int occupies_memory;
};//}}}

struct Box
// Cubical Box that holds float values
{//{{{
    Box(const long long int Nside, const float sidelength)
    {//{{{
        this->Nside = Nside;
        this->sidelength = sidelength;
        this->a = this->sidelength/(float)(this->Nside);
        try
        {
            this->values = new float[this->Nside*this->Nside*this->Nside];
        }
        catch (exception& e)
        {
            cout << "Standard exception :" << e.what() << endl;
        }
        // initialize to zero
        for (int ii=0; ii<this->Nside*this->Nside*this->Nside; ii++) { this->values[ii] = 0.0; }
    }//}}}
    ~Box()
    {//{{{
        delete[] this->values;
    }//}}}
    void index_at_position(const float X, const float Y, const float Z, long long int *xx, long long int *yy, long long int *zz)
    {//{{{
        *xx = (long long int)(X/this->a); // round down
        *yy = (long long int)(Y/this->a); // round down
        *zz = (long long int)(Z/this->a); // round down
    }//}}}
    void add(const float v, const long long int xx, const long long int yy, const long long int zz)
    {//{{{
        #pragma omp atomic
        this->values[
            this->Nside*this->Nside*((this->Nside+xx%this->Nside)%this->Nside)
            + this->Nside          *((this->Nside+yy%this->Nside)%this->Nside)
            +                       ((this->Nside+zz%this->Nside)%this->Nside)
            ] += v;
    }//}}}
    void multiply(const float v, const long long int xx, const long long int yy, const long long int zz)
    {//{{{
        #pragma omp atomic
        this->values[
            this->Nside*this->Nside*((this->Nside+xx%this->Nside)%this->Nside)
            + this->Nside          *((this->Nside+yy%this->Nside)%this->Nside)
            +                       ((this->Nside+zz%this->Nside)%this->Nside)
            ] *= v;
    }//}}}
    void set(const float v, const long long int xx, const long long int yy, const long long int zz)
    {//{{{
        this->values[
            this->Nside*this->Nside*((this->Nside+xx%this->Nside)%this->Nside)
            + this->Nside          *((this->Nside+yy%this->Nside)%this->Nside)
            +                       ((this->Nside+zz%this->Nside)%this->Nside)
            ] = v;
    }//}}}
    float get(const long long int xx, const long long int yy, const long long int zz)
    {//{{{
        return this->values[
            this->Nside*this->Nside*((this->Nside+xx%this->Nside)%this->Nside)
            + this->Nside          *((this->Nside+yy%this->Nside)%this->Nside)
            +                       ((this->Nside+zz%this->Nside)%this->Nside)
            ];
    }//}}}
    void fill_box(const long int Nspheres, const float *sphere_centres,
                  const float *sphere_radii, const float *sphere_weights)
    {//{{{
        #pragma omp parallel for
        for (long int ii=0; ii<Nspheres; ii++)
        {
            long long int xx_min, yy_min, zz_min;
            long long int xx_max, yy_max, zz_max;
            index_at_position(sphere_centres[3*ii+0]+sphere_radii[ii],
                              sphere_centres[3*ii+1]+sphere_radii[ii],
                              sphere_centres[3*ii+2]+sphere_radii[ii],
                              &xx_max, &yy_max, &zz_max);
            index_at_position(sphere_centres[3*ii+0]-sphere_radii[ii],
                              sphere_centres[3*ii+1]-sphere_radii[ii],
                              sphere_centres[3*ii+2]-sphere_radii[ii],
                              &xx_min, &yy_min, &zz_min);
            Sphere s{{sphere_centres[3*ii], sphere_centres[3*ii+1], sphere_centres[3*ii+2]},
                     sphere_radii[ii]};
            for (long long int xx=xx_min; xx<=xx_max; xx++)
            {
                for (long long int yy=yy_min; yy<=yy_max; yy++)
                {
                    for (long long int zz=zz_min; zz<=zz_max; zz++)
                    {
                        //{{{
                        vector_t v0{
                            (float)(xx)*this->a,
                            (float)(yy)*this->a,
                            (float)(zz)*this->a,
                            };
                        vector_t v1{
                            (float)(xx + 1)*this->a,
                            (float)(yy)*this->a,
                            (float)(zz)*this->a,
                            };
                        vector_t v2{
                            (float)(xx + 1)*this->a,
                            (float)(yy + 1)*this->a,
                            (float)(zz)*this->a,
                            };
                        vector_t v3{
                            (float)(xx)*this->a,
                            (float)(yy + 1)*this->a,
                            (float)(zz)*this->a,
                            };
                        vector_t v4{
                            (float)(xx)*this->a,
                            (float)(yy)*this->a,
                            (float)(zz + 1)*this->a,
                            };
                        vector_t v5{
                            (float)(xx + 1)*this->a,
                            (float)(yy)*this->a,
                            (float)(zz + 1)*this->a,
                            };
                        vector_t v6{
                            (float)(xx + 1)*this->a,
                            (float)(yy + 1)*this->a,
                            (float)(zz + 1)*this->a,
                            };
                        vector_t v7{
                            (float)(xx)*this->a,
                            (float)(yy + 1)*this->a,
                            (float)(zz + 1)*this->a,
                            };
                        //}}}
                        Hexahedron hex{v0,v1,v2,v3,v4,v5,v6,v7};
                        this->add(overlap(s,hex)*sphere_weights[ii]/this->a/this->a/this->a,
                                  xx, yy, zz);
                    }
                }
            }
        }
    }//}}}
    void save_to_file(string name, long long int fraction)
    // produces fraction^3 individual files, each is a separate cube
    {//{{{
        long long int L = this->Nside/fraction;
        for (long long int xchunk=0; xchunk<fraction; xchunk++)
        {
            for (long long int ychunk=0; ychunk<fraction; ychunk++)
            {
                for (long long int zchunk=0; zchunk<fraction; zchunk++)
                {
                    FILE *f = fopen((name+to_string(xchunk)+"_"+to_string(ychunk)+"_"+to_string(zchunk)).c_str(), "wb");
                    for (long long int xx=xchunk*L; xx<(xchunk+1)*L; xx++)
                    {
                        for (long long int yy=ychunk*L; yy<(ychunk+1)*L; yy++)
                        {
                            fwrite(this->values+xx*this->Nside*this->Nside+yy*this->Nside+zchunk*L, sizeof(float), L, f);
                        }
                    }
                    fclose(f);
                }
            }
        }
    }//}}}
    long long int Nside;
    float *values;
    float sidelength;
    float a;
};//}}}

void box_filling(Box *b, string INPUT_PREFIX, int PTYPE, int NCHUNKS, int axis = -1)
{//{{{
    if ((PTYPE==MOMDM) || (PTYPE==VELDM) || (PTYPE==MOMGAS))
    {
        if ((axis < 0) || (axis > 2))
        {
            cout << "Invalid momentum axis " << axis << endl;
            return;
        }
    }

    // loop over manageable chunks
    for (int chunk=0; chunk<NCHUNKS; chunk++)
    {
        cout << "Chunk Nr " << (chunk+1) << " out of " << NCHUNKS << endl;
        SimChunk *s;
        s = new SimChunk(INPUT_PREFIX + to_string(chunk) + ".hdf5");
        
        Field coords = Field(s, PTYPE, "Coordinates");
        coords.read_to_memory();

        long int Nparticles = coords.dim_lengths[0];
        float *radii = new float[Nparticles];
        float *weight = new float[Nparticles];

        // compute radii and weight
        if (PTYPE==GAS)
        {
            Field dens = Field(s, PTYPE, "Density");
            dens.read_to_memory();
            Field mass = Field(s, PTYPE, "Masses");
            mass.read_to_memory();
            Field internal_energy = Field(s, PTYPE, "InternalEnergy");
            internal_energy.read_to_memory();
            Field electron_abundance = Field(s, PTYPE, "ElectronAbundance");
            electron_abundance.read_to_memory();
            for (int ii=0; ii<Nparticles; ii++)
            {
                // radius of the approximately spherical Voronoi cell
                radii[ii] = std::cbrt(3.0*mass[ii]/dens[ii]/4.0/M_PI);

                // Eqs 7, 8 from Komatsu & Seljak 2002 arxiv.org/pdf/astro-ph/0205468.pdf
                weight[ii] = (2.0*(1.0+XH)/(1.0+3.0*XH+4.0*XH*electron_abundance[ii])
                              *(GAMMA-1.0)*dens[ii]*internal_energy[ii]);
                // this is in (1e10 Msun/h)/(ckpc/h)^3 * (km/s)^2
            }
        }
        else if (PTYPE==NE)
        {
            Field dens = Field(s, PTYPE, "Density");
            dens.read_to_memory();
            Field mass = Field(s, PTYPE, "Masses");
            mass.read_to_memory();
            Field electron_abundance = Field(s, PTYPE, "ElectronAbundance");
            electron_abundance.read_to_memory();
            for (int ii=0; ii<Nparticles; ii++)
            {
                // radius of the approximately spherical Voronoi cell
                radii[ii] = std::cbrt(3.0*mass[ii]/dens[ii]/4.0/M_PI);

                // ne = XH * rho / mp * ElectronAbundance
                weight[ii] = XH * dens[ii] * electron_abundance[ii] / MPROTON;
                // this is in (1e10 Msun/h)/(ckpc/h)^3
            }
        }
        else if (PTYPE==MOMGAS)
        {
            Field dens = Field(s, PTYPE, "Density");
            dens.read_to_memory();
            Field mass = Field(s, PTYPE, "Masses");
            mass.read_to_memory();
            Field electron_abundance = Field(s, PTYPE, "ElectronAbundance");
            electron_abundance.read_to_memory();
            Field velocity = Field(s, PTYPE, "Velocities");
            velocity.read_to_memory();
            for (int ii=0; ii<Nparticles; ii++)
            {
                // radius of the approximately spherical Voronoi cell
                radii[ii] = std::cbrt(3.0*mass[ii]/dens[ii]/4.0/M_PI);

                // ne = XH * rho / mp * ElectronAbundance
                weight[ii] = XH * dens[ii] * electron_abundance[ii] / MPROTON * velocity[3*ii+axis];
                // this is in (1e10 Msun/h)/(ckpc/h)^3 * km/s
            }
        }
        else if (PTYPE==DM)
        {
            Field dens = Field(s, PTYPE, "SubfindDensity"); // density smoothed over Hsml sphere
            dens.read_to_memory();
            Field hsml = Field(s, PTYPE, "SubfindHsml"); // radius enclosing the nearest 64 DM particles
            hsml.read_to_memory();
            for (int ii=0; ii<Nparticles; ii++)
            {
                radii[ii] = hsml[ii]/4.0f;
                weight[ii] = dens[ii];
            }
        }
        else if (PTYPE==MOMDM)
        {
            Field dens = Field(s, PTYPE, "SubfindDensity"); // density smoothed over Hsml sphere
            dens.read_to_memory();
            Field hsml = Field(s, PTYPE, "SubfindHsml"); // radius enclosing the nearest 64 DM particles
            hsml.read_to_memory();
            Field velocity = Field(s, PTYPE, "Velocities");
            velocity.read_to_memory();
            for (int ii=0; ii<Nparticles; ii++)
            {
                radii[ii] = hsml[ii]/4.0f;
                weight[ii] = dens[ii] * velocity[3*ii+axis];
            }
        }
        else if (PTYPE==VELDM)
        {
            Field hsml = Field(s, PTYPE, "SubfindHsml"); // radius enclosing the nearest 64 DM particles
            hsml.read_to_memory();
            Field velocity = Field(s, PTYPE, "Velocities");
            velocity.read_to_memory();
            for (int ii=0; ii<Nparticles; ii++)
            {
                radii[ii] = hsml[ii]/4.0f;
                weight[ii] = velocity[3*ii+axis];
            }
        }

        // Now fill the box with the particles in the current chunk
        b->fill_box(Nparticles, coords.values, radii, weight);
        
        delete   s;
        delete[] radii;
        delete[] weight;
    }
}//}}}

int main(int argc, char **argv)
{//{{{

    if (argc!=3) { cout << "wrong number of arguments." << endl; return -1; }

    cout << "Going to use " << std::getenv("OMP_NUM_THREADS") << " threads." << endl;

    const int PTYPE = atoi(argv[1]);
    const string INPUT_PREFIX(argv[2]);
    const string OUTPUT_PREFIX(argv[3]);
    const int NCHUNKS = atoi(argv[4]);
    const float BOX_SIZE = atof(argv[5]);
    const long long int NSIDE = atoll(argv[6]); 
    const int NSUBBOXES = atoi(argv[7]);
    const int AXIS = ((PTYPE==MOMDM) || (PTYPE==VELDM) || (PTYPE==MOMGAS))
                     ? atoi(argv[8]) : -1;

    Box *b = new Box(NSIDE, BOX_SIZE);

    box_filling(b, INPUT_PREFIX, PTYPE, NCHUNKS, AXIS);

    b->save_to_file(OUTPUT_PREFIX, NSUBBOXES);

    delete b;
    
    return 0;
}//}}}
