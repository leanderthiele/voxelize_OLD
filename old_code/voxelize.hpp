#ifndef VOXELIZE_H
#define VOXELIZE_H

#include <string>
#include <array>
#include <valarray>

#include <H5Cpp.h>

#include "overlap_lft.hpp"

class SimChunk : public H5::H5File
{//{{{
    public :
        SimChunk(const std::string fname); // constructor

        template <typename T> T get_header_field(const std::string name) const;
};//}}}

class Field : public std::valarray<float>
{//{{{
    public :
        Field(const SimChunk& s, const int ptype, const std::string name);
        ~Field();

        // returns number of particles that have been read
        long read_to_memory(const long Nparticles=1000000L);

        void clear_memory();

        // get element safely
        float at(const long i) const;
        // convenience function for vectors
        float at(const long i1, const int i2) const;

        long length;
        int Ndims;
    
    private :

        long pos; // current zeroth particle

        H5::DataSet   *dset;
        H5::DataSpace *dspace;
        int rank;
        hsize_t dim_lengths[10];
        int per_particle;
};//}}}

class Box : public std::valarray<float>
{//{{{
    public :
        Box(const long N, const float L, int dim=1);

        void fill_box(const Field& centres,
                      const Field& radii,
                      const std::valarray<float>& weights);
        void save(const std::string name) const;

    private :
        inline void index_at_pos(const float x, const float y, const float z,
                                 long& xx, long& yy, long& zz) const;
        
        inline long three_to_one(const long x, const long y, const long z) const;
        inline long four_to_one(const long x, const long y, const long z, const int dd) const;

        long N;
        float L;
        float a;
        int dim;
};//}}}

class Formula
{//{{{
    public :
        Formula(const std::string eq);

        int next_field(std::string& name, float& exponent);

        float prefactor;

    private :
        int current_field;
        std::vector<std::pair<std::string, float>> split_eq;
};//}}}

#endif // VOXELIZE_H
