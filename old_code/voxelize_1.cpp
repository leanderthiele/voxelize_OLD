#include <cmath>
#include <iostream>

#include "voxelize.hpp"

#include "overlap_lft.hpp"

// SimChunk {{{
SimChunk::SimChunk(const std::string fname) :
    H5::H5File(fname, H5F_ACC_RDONLY) { }
template <typename T> T SimChunk::get_header_field(const std::string name) const
{
    H5::Group header(this->openGroup("/Header"));
    H5::Attribute attr = header.openAttribute(name);
    T out;
    attr.read(attr.getDataType(), &out);
    return out;
}
//}}}

// Field {{{
Field::Field(const SimChunk& s, const int ptype, const std::string name) :
    pos {0L}, length {0L}
{
    dset = new H5::DataSet(s.openDataSet("PartType"
                           + std::to_string(ptype)
                           + "/" + name));

    dspace = new H5::DataSpace(dset->getSpace());
    rank = dspace->getSimpleExtentNdims();
    Ndims = dspace->getSimpleExtentDims(dim_lengths);

    per_particle = 1;
    for (int ii=1; ii<Ndims; ii++)
        per_particle *= dim_lengths[ii];
}
Field::~Field()
{
    delete dspace;
    delete dset;
}
long Field::read_to_memory(const long Nparticles)
{
    long remains = std::min(Nparticles, (long)dim_lengths[0] - pos);
    if (remains == 0)
        throw 1;

    hsize_t these_dim_lengths[10];
    these_dim_lengths[0] = remains;
    for (int ii=1; ii<Ndims; ii++)
        these_dim_lengths[ii] = dim_lengths[ii];

    H5::DataSpace memspace(Ndims, these_dim_lengths);
    this->resize(memspace.getSimpleExtentNpoints());

    hsize_t offset[10];
    offset[0] = pos;
    for (int ii=1; ii<Ndims; ii++)
        offset[ii] = 0;

    dspace->selectHyperslab(H5S_SELECT_SET, these_dim_lengths, offset);

    dset->read(&((*this)[0]), H5::PredType::NATIVE_FLOAT, memspace, *dspace);

    pos += these_dim_lengths[0];
    length = these_dim_lengths[0];

    return these_dim_lengths[0];
}
void Field::clear_memory()
{
    this->resize(0);
}
float Field::at(const long i) const
{
    if (i >= pos * (long)per_particle)
        throw 1;

    return (*this)[i - pos*(long)per_particle];
}
float Field::at(const long i1, const int i2) const
{
    return this->at(i1*dim_lengths[1]+i2);
}
//}}}

// Box {{{
Box::Box(const long N, const float L, const int dim) :
    std::valarray<float>(0.0f, N*N*N*dim),
    N {N}, L {L}, dim {dim}, a {L/(float)N} { }
void Box::fill_box(const Field& centres,
                   const Field& radii,
                   const std::valarray<float>& weights)
{
    #pragma omp parallel for
    for (long ii=0; ii<centres.length; ii++)
    {
        long x_min, y_min, z_min, x_max, y_max, z_max;
        index_at_pos(centres.at(ii,0) + radii[ii],
                     centres.at(ii,1) + radii[ii],
                     centres.at(ii,2) + radii[ii],
                     x_max, y_max, z_max);
        index_at_pos(centres.at(ii,0) - radii[ii],
                     centres.at(ii,1) - radii[ii],
                     centres.at(ii,2) - radii[ii],
                     x_min, y_min, z_min);
        
        Olap::Sphere sph{{centres.at(ii,0), centres.at(ii,1), centres.at(ii,2)},
                         radii[ii]};

        for (long xx=x_min; xx<=x_max; xx++)
        {
            for (long yy=y_min; yy<=y_max; yy++)
            {
                for (long zz=z_min; zz<=z_max; zz++)
                {
                    Olap::vector_t v0{ (float)(xx)    *a,
                                       (float)(yy)    *a,
                                       (float)(zz)    *a, };
                    Olap::vector_t v1{ (float)(xx + 1)*a,
                                       (float)(yy)    *a,
                                       (float)(zz)    *a, };
                    Olap::vector_t v2{ (float)(xx + 1)*a,
                                       (float)(yy + 1)*a,
                                       (float)(zz)    *a, };
                    Olap::vector_t v3{ (float)(xx)    *a,
                                       (float)(yy + 1)*a,
                                       (float)(zz)    *a, };
                    Olap::vector_t v4{ (float)(xx)    *a,
                                       (float)(yy)    *a,
                                       (float)(zz + 1)*a, };
                    Olap::vector_t v5{ (float)(xx + 1)*a,
                                       (float)(yy)    *a,
                                       (float)(zz + 1)*a, };
                    Olap::vector_t v6{ (float)(xx + 1)*a,
                                       (float)(yy + 1)*a,
                                       (float)(zz + 1)*a, };
                    Olap::vector_t v7{ (float)(xx)    *a,
                                       (float)(yy + 1)*a,
                                       (float)(zz + 1)*a, };

                    Olap::Hexahedron hex{v0,v1,v2,v3,v4,v5,v6,v7};

                    float tmp = Olap::overlap(sph, hex)/(a * a * a);

                    if (this->dim != 1)
                    {
                        for (int dd=0; dd<dim; dd++)
                        {
                            (*this)[four_to_one(xx,yy,zz,dd)]
                                += tmp * weights[ii*dim+dd];
                        }
                    }
                    else
                    {
                        (*this)[three_to_one(xx,yy,zz)]
                            += tmp * weights[ii];
                    }
                }
            }
        }
    }
}
void Box::save(const std::string fname) const
{
    H5::H5File f(fname, H5F_ACC_TRUNC);
    
    int frank;
    hsize_t fdim[10];
    for (int ii=0; ii<3; ii++)
        fdim[ii] = N;
    if (dim != 1)
    {
        fdim[4] = dim;
        frank = 4;
    }
    else
    {
        frank = 3;
    }

    H5::DataSpace fspace(frank, fdim);

    H5::DataSet ds(f.createDataSet("data", H5::PredType::NATIVE_FLOAT, fspace));

    ds.write(&((*this)[0]), H5::PredType::NATIVE_FLOAT);
}
inline void Box::index_at_pos(const float x, const float y, const float z,
                              long& xx, long& yy, long& zz) const
{
    xx = (long)(x/a); yy = (long)(y/a); zz = (long)(z/a);
}
inline long Box::three_to_one(const long x, const long y, const long z) const
{
    return N * N * ((N+x%N)%N)
           +   N * ((N+y%N)%N)
           +       ((N+z%N)%N);
}
inline long Box::four_to_one(const long x, const long y, const long z, const int dd) const
{
    return three_to_one(x,y,z) * (long)(dim) + (long)dd;
}
//}}}

// Formula {{{
Formula::Formula(const std::string eq) :
    current_field {0}, prefactor {1.0f}
{
    // split the equation along * and /
    int current_mode = 1; // the first value will always have an implicit *

    std::string eq1(eq);
    while (1)
    {
        size_t pos_star  = eq1.find('*');
        size_t pos_slash = eq1.find('/');
        size_t pos = std::min(pos_star, pos_slash);

        std::string seg(eq1.substr(0, pos));

        eq1.erase(0, pos+1);

        size_t pos_hat = seg.find('^');
        float exponent;
        if (pos_hat == std::string::npos)
            exponent = (float)current_mode;
        else
            exponent = (float)(current_mode) * stof(seg.substr(pos_hat+1));

        try
        {
            float base = stof(seg.substr(0, pos_hat-1));
            prefactor *= std::pow(base, exponent);
        }
        catch (const std::invalid_argument& ia)
        {
            split_eq.push_back(std::pair<std::string, float>(seg.substr(0, pos_hat),
                                                             exponent));
        }

        if (pos_star < pos_slash)
            current_mode = 1;
        else if (pos_slash < pos_star)
            current_mode = -1;
        else
            break;
    }
}
int Formula::next_field(std::string& name, float& exponent)
{
    if (current_field < split_eq.size())
    {
        name = split_eq.at(current_field).first;
        exponent = split_eq.at(current_field).second;
        ++current_field;
        return 0;
    }
    else
    {
        return 1;
    }
}
//}}}


int main(int argc, char **argv)
{
    const std::string eq(argv[1]);

    Formula f(eq);

    std::string tmp;
    float e;
    while (!(f.next_field(tmp, e)))
    {
        std::cout << tmp << "^" << e << "\n";
    }

    return 0;
}
