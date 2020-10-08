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
#include "/home/lthiele/Overlaps/overlap/overlap_lft.hpp"

// Physical constants
#define KBOLTZM (1.3806485e-16)        // cgs
#define GNEWTON (4.3e4f)               // (kpc/1e10 Msun) (km/s)^2
#define XH      (0.76)                 // hydrogen mass fraction, dimensionless
#define GAMMA   (5.0/3.0)              // adiabatic index, dimensionless
#define MSUN    (1.98847e43*0.6774)    // cgs (1e10 Msun/h)
#define MPROTON (1836.15267343)        // in electron masses

// particle types
#define GAS 0 // electron pressure
#define DM  1 // DM density
#define NE  2 // electron number density
#define MOMDM 3 // dark matter momentum
#define MOMGAS 4 // electron momentum density
#define VELDM 5 // dark matter velocity

// mode types
#define FILL 0
#define PAINT 1

#define MMIN 1000.0f // minimum halo mass to consider (in 1e10 Msun/h)
#define ROUT_SCALE 3.0f // in terms of Rvir

// box size
#define RESOLUTION 1024
#define NSUBBOXES  2

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
        int Nvalues = 1;
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
    float &operator[] (const int index) const
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

struct Cosmology
{//{{{
    Cosmology(const float OM, const float OL, const float OB, const float h)
    {
        this->OM = OM;
        this->OL = OL;
        this->OB = OB;
        this->h  = h;
    }
    float rho_c (const float a) const
    {
        return 2.775e-8f * (this->OM/a/a/a + this->OL);
        // 1e10 Msun/h / (kpc/h)^3
    }
    float OM;
    float OL;
    float OB;
    float h;
};//}}}

struct Halo
// A single halo
{//{{{
    Halo(const float *Mvir, const float *Rvir, const double *a)
    {
        // NFW + Duffy 08
        float cvir = 7.85f * powf(*Mvir/200.0f, 0.081f) * powf((float)(*a), 0.71f);
        this->Rs   = *Rvir/cvir;
        this->rho0 = *Mvir/4.0f/M_PI/this->Rs/this->Rs/this->Rs/(log1p(cvir) - cvir/(1.0f+cvir));
        this->Rout = ROUT_SCALE * *Rvir;
        this->profile = &Halo::density;
        this->M = *Mvir;
    }
    Halo(const Cosmology *C, const float *Mvir, const float *M200c, const float *R200c, const float *Rvir, const double *a)
    {
        // Battaglia+2012
        this->P200c = 100.0f * GNEWTON * *M200c * C->rho_c((float)(*a)) * C->OB / C->OM / *R200c;
        this->Pi0   = 18.1f  * powf((*M200c/1e4f/C->h), 0.154f)    * powf((float)(*a), 0.758f);
        this->Rc    = 0.497f * powf((*M200c/1e4f/C->h), -0.00865f) * powf((float)(*a), -0.731f) * *R200c;
        this->beta  = 4.35f  * powf((*M200c/1e4f/C->h), 0.0393f)   * powf((float)(*a), -0.415f);
        this->Rout  = ROUT_SCALE * *Rvir;
        this->profile = &Halo::pressure;
        this->M = *Mvir;
    }
    float density(float r) const
    {
        if (r > this->Rout) { return 0.0f; }
        return this->rho0/(r/this->Rs)/(1.0f+r/this->Rs)/(1.0f+r/this->Rs);
    }
    float pressure(float r) const
    {
        if (r > this->Rout) { return 0.0f; }
        return this->P200c*this->Pi0*powf(r/this->Rc, -0.3f)/powf(1.0f+r/this->Rc, this->beta);
    }
    float M;
    float Rout;
    float Rs;
    float rho0;
    float P200c;
    float Pi0;
    float Rc;
    float beta;
    float (Halo::*profile)(float) const;
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
    void fill_box(const int Nspheres, const float *sphere_centres,
              const float *sphere_radii, const float *sphere_weights)
    {//{{{
        #pragma omp parallel for
        for (int ii=0; ii<Nspheres; ii++)
        {
            long long int xx_min, yy_min, zz_min;
            long long int xx_max, yy_max, zz_max;
//          if ((ii+1)%1000000 == 0)
//          {
//              cout << (ii+1)/1000000 << " out of " << (int)ceil(Nspheres/1000000) << endl;
//          }
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
    void paint_halos(const int Nhalos, const float *halo_centres, Halo **h, const int Nsample, const float Mmin)
    {//{{{
        #pragma omp parallel for
        for (int ii=0; ii<Nhalos; ii++)
        {
            if (h[ii]->M < Mmin) { continue; }
            long long int xx_min, yy_min, zz_min;
            long long int xx_max, yy_max, zz_max;
            index_at_position(halo_centres[3*ii+0]+h[ii]->Rout,
                              halo_centres[3*ii+1]+h[ii]->Rout,
                              halo_centres[3*ii+2]+h[ii]->Rout,
                              &xx_max, &yy_max, &zz_max);
            index_at_position(halo_centres[3*ii+0]-h[ii]->Rout,
                              halo_centres[3*ii+1]-h[ii]->Rout,
                              halo_centres[3*ii+2]-h[ii]->Rout,
                              &xx_min, &yy_min, &zz_min);
            for (long long int xx=xx_min; xx<=xx_max; xx++)
            {
                for (long long int yy=yy_min; yy<=yy_max; yy++)
                {
                    for (long long int zz=zz_min; zz<=zz_max; zz++)
                    {
                        float this_value = 0.0f;
                        for (int small_xx=-Nsample; small_xx<=Nsample; small_xx++)
                        {
                            for (int small_yy=-Nsample; small_yy<=Nsample; small_yy++)
                            {
                                for (int small_zz=-Nsample; small_zz<=Nsample; small_zz++)
                                {
                                    float xpos = ((float)(xx)+0.5f+(float)(small_xx)/(float)(2*Nsample+1))*this->a;
                                    float ypos = ((float)(yy)+0.5f+(float)(small_yy)/(float)(2*Nsample+1))*this->a;
                                    float zpos = ((float)(zz)+0.5f+(float)(small_zz)/(float)(2*Nsample+1))*this->a;
                                    float r = std::sqrt(
                                            (halo_centres[3*ii+0]-xpos)*(halo_centres[3*ii+0]-xpos)
                                            + (halo_centres[3*ii+1]-ypos)*(halo_centres[3*ii+1]-ypos)
                                            + (halo_centres[3*ii+2]-zpos)*(halo_centres[3*ii+2]-zpos)
                                            );
                                    this_value += ((h[ii]->*(h[ii]->profile))(r)
                                                   /(float)((2*Nsample+1)*(2*Nsample+1)*(2*Nsample+1)));
                                }
                            }
                        }
                        if (this_value > this->get(xx, yy, zz))
                        {
                            this->set(this_value, xx, yy, zz);
                        }
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

void box_painting(Box *b, int PTYPE, int Nchunks)
{//{{{
    Cosmology *C;
    C = new Cosmology(0.3086, 0.6911, 0.0486, 0.6774);

    for (int chunk=0; chunk<Nchunks; chunk++)
    {
        cout << "Chunk Nr " << (chunk+1) << " out of " << Nchunks << endl;
        SimChunk *g;
        if (PTYPE == GAS)
        {
            g = new SimChunk("/tigress/lthiele/Illustris_300-1_TNG/output/groups_099/fof_subhalo_tab_099."+to_string(chunk)+".hdf5");
        }
        else if (PTYPE == DM)
        {
            g = new SimChunk("/tigress/lthiele/Illustris_300-1_Dark/output/groups_099/fof_subhalo_tab_099."+to_string(chunk)+".hdf5");
        }
        else
        {
            cout << "Unsupported particle type for mode FILL. Aborting.\n" << endl;
            exit(-1);
        }
        int Ngroups;
        g->get_header_field("Ngroups_ThisFile", &Ngroups);
        if (Ngroups == 0)
        {
            delete g;
            continue;
        }
        double a;
        g->get_header_field("Time", &a);
        Field coords = Field(g, "Group/GroupPos");
        coords.read_to_memory();
        Halo **h;
        h = new Halo*[Ngroups];
        if (PTYPE == GAS)
        {
            Field Mvir = Field(g, "Group/Group_M_TopHat200");
            Mvir.read_to_memory();
            Field M200c = Field(g, "Group/Group_M_Crit200");
            M200c.read_to_memory();
            Field R200c = Field(g, "Group/Group_R_Crit200");
            R200c.read_to_memory();
            Field Rvir = Field(g, "Group/Group_R_TopHat200");
            Rvir.read_to_memory();
            // TODO comoving vs physical --> Halo takes physical I believe
            for (int ii=0; ii<Ngroups; ii++)
            {
                h[ii] = new Halo(C, Mvir.values+ii, M200c.values+ii, R200c.values+ii, Rvir.values+ii, &a);
            }
        }
        else if (PTYPE == DM)
        {
            Field Mvir = Field(g, "Group/Group_M_TopHat200");
            Mvir.read_to_memory();
            Field Rvir = Field(g, "Group/Group_R_TopHat200");
            Rvir.read_to_memory();
            for (int ii=0; ii<Ngroups; ii++)
            {
                h[ii] = new Halo(Mvir.values+ii, Rvir.values+ii, &a);
            }
        }

        b->paint_halos(Ngroups, coords.values, h, 1, MMIN);
        // paint only halos with Mvir > 1e8 Msun/h

        for (int ii=0; ii<Ngroups; ii++)
        {
            delete h[ii];
        }
        delete[] h;
        delete g;
    }
}//}}}

void box_filling(Box *b, int PTYPE, int Nchunks, int axis = -1)
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
    for (int chunk=0; chunk<Nchunks; chunk++)
    {
        cout << "Chunk Nr " << (chunk+1) << " out of " << Nchunks << endl;
        SimChunk *s;
        if ((PTYPE==GAS) || (PTYPE==NE) || (PTYPE==MOMGAS))
        {
            s = new SimChunk("/tigress/lthiele/Illustris_300-1_TNG/output/snapdir_099/snap_099."+to_string(chunk)+".hdf5");
        }
        else if ((PTYPE==DM) || (PTYPE==MOMDM) || (PTYPE==VELDM))
        {
            s = new SimChunk("/tigress/lthiele/Illustris_300-1_Dark/output/snapdir_099/snap_099."+to_string(chunk)+".hdf5");
        }
        
        Field coords = Field(s, PTYPE, "Coordinates");
        coords.read_to_memory();

        int Nparticles = coords.dim_lengths[0];
        float *radii = new float[Nparticles];
        float *weight = new float[Nparticles]; // for GAS: electron pressure, for DM: density

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
    const int MODE  = atoi(argv[2]);
    int Nchunks;
    if (PTYPE==GAS)
    {
        cout << "Working with GAS." << endl;
        Nchunks = 600;
    }
    else if (PTYPE==DM)
    {
        cout << "Working with DM." << endl;
        Nchunks = 75;
    }
    else if (PTYPE==NE)
    {
        cout << "Working with NE." << endl;
        Nchunks = 600;
    }
    else if (PTYPE==MOMDM)
    {
        cout << "Working with DM momentum density." << endl;
        Nchunks = 75;
    }
    else if (PTYPE==VELDM)
    {
        cout << "Working with DM velocity." << endl;
        Nchunks = 75;
    }
    else if (PTYPE==MOMGAS)
    {
        cout << "Working with GAS momentum density." << endl;
        Nchunks = 600;
    }
    else
    {
        cout << "Ptype " << PTYPE << " not supported." << endl;
        return -1;
    }

    const float BoxSize = 205000.0;
    Box *b = new Box(RESOLUTION, BoxSize);

    if ((PTYPE == MOMDM) || (PTYPE == VELDM) ||(PTYPE == MOMGAS))
    {
        cout << "In mode FILL for momentum density." << endl;
        box_filling(b, PTYPE, Nchunks, MODE);
    }
    else
    {
        if (MODE == FILL)
        {
            cout << "In mode FILL." << endl;
            box_filling(b, PTYPE, Nchunks);
        }
        else if (MODE == PAINT)
        {
            cout << "In mode PAINT." << endl;
            box_painting(b, PTYPE, Nchunks);
        }
        else
        {
            cout << "Invalid mode " << MODE << endl;
            return -1;
        }
    }

    if (PTYPE == GAS)
    {
        if (MODE == FILL)
        {
            b->save_to_file("/scratch/gpfs/lthiele/gas_boxes_"+to_string(RESOLUTION)+"/TEST_box_", NSUBBOXES);
        }
        else if (MODE == PAINT)
        {
            b->save_to_file("/tigress/lthiele/boxes/painted_gas_boxes_"+to_string(RESOLUTION)+"/TEST_box_", NSUBBOXES);
        }
    }
    else if (PTYPE == DM)
    {
        if (MODE == FILL)
        {
            b->save_to_file("/scratch/gpfs/lthiele/DM_boxes_"+to_string(RESOLUTION)+"/TEST_box_", NSUBBOXES);
        }
        else if (MODE == PAINT)
        {
            b->save_to_file("/tigress/lthiele/boxes/painted_DM_boxes_"+to_string(RESOLUTION)+"/TEST_box_", NSUBBOXES);
        }
    }
    else if (PTYPE == NE)
    {
        if (MODE == FILL)
        {
            b->save_to_file("/tigress/lthiele/boxes/NE_boxes_"+to_string(RESOLUTION)+"/TEST_box_", NSUBBOXES);
        }
    }
    else if (PTYPE == MOMDM)
    {
        b->save_to_file("/tigress/lthiele/boxes/MOMDM_boxes_"+to_string(RESOLUTION)+"_"+to_string(MODE)+"/TEST_box_", NSUBBOXES);
    }
    else if (PTYPE == VELDM)
    {
        b->save_to_file("/tigress/lthiele/boxes/VELDM_boxes_"+to_string(RESOLUTION)+"_"+to_string(MODE)+"/TEST_box_", NSUBBOXES);
    }
    else if (PTYPE == MOMGAS)
    {
        b->save_to_file("/tigress/lthiele/boxes/MOMGAS_boxes_"+to_string(RESOLUTION)+"_"+to_string(MODE)+"/TEST_box_", NSUBBOXES);
    }
    delete b;
    
    return 0;
}//}}}
