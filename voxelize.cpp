#include <iostream>
#include <array>
#include <algorithm>
#include <vector>
#include <cmath>

#include <omp.h>

#include <torch/torch.h>

#include "overlap_lft_double.hpp"
#include "network.hpp"
#include "voxelize.hpp"

template<int batch_size>
struct Interpolator
{//{{{
    Interpolator (const std::string &net_fname) :
        input    { torch::empty( {batch_size, 8}, torch::kFloat32 ) },
        output   { torch::empty( {batch_size, 1}, torch::kFloat32 ) },
        input_a  { input.accessor<float,2>() },
        output_a { output.accessor<float,2>() }
    {
        torch::load(net, net_fname);
    }

    bool is_full() const { return num_points == batch_size; }

    void add_point(long _idx, long _pidx, float R, std::array<float,3> &cub)
    {
        idx.push_back(_idx);
        pidx.push_back(_pidx);
        vol.push_back( std::min(4.188790204786*R*R*R, 1.0) );
        
        // the original values
        input_a[num_points][0] = R;
        for (int ii = 0; ii != 3; ++ii)
            input_a[num_points][ii+1] = cub[ii];

        // rescaled values
        input_a[num_points][4] = std::log(R);
        for (int ii = 0; ii != 3; ++ii)
            input_a[num_points][ii+5] = cub[ii] / R;

        // update the counter
        ++num_points;
    }
    
    torch::TensorAccessor<float,2> &result ()
    {
        output = net->forward(input);
        output_a = output.accessor<float,2>();

        // neutralize the internal network normalization
        for (int ii = 0; ii != vol.size(); ++ii)
            output_a[ii][0] *= vol[ii];

        return output_a;
    }

    void reset()
    {
        // passing the input tensor through the network alters its shape!
        input = torch::empty( {batch_size, 8}, torch::kFloat32 );

        // we also need to reset the accessor
        input_a = input.accessor<float,2>();

        // re-initialize the vectors
        idx.clear(); pidx.clear(); vol.clear();
        num_points = 0;
    }

    std::shared_ptr<Net<8,8,64>> net = std::make_shared<Net<8,8,64>>();

    std::vector<long> idx;
    std::vector<long> pidx;
    std::vector<float> vol;
    
    torch::Tensor input;
    torch::Tensor output;
    torch::TensorAccessor<float,2> input_a;
    torch::TensorAccessor<float,2> output_a;

    int num_points = 0;

    constexpr static float Rmin = 1e-2;
    constexpr static float Rmax = 1e2;
};//}}}

static inline float
sph_overlap_exact (const float R,
                   const std::array<float,3> &cub)
// assumes the cube coordinates have been normalized
{//{{{
    Olap::Sphere sph { {0.0F, 0.0F, 0.0F}, R };

    Olap::vector_t v0 {cub[0],      cub[1],      cub[2]     };
    Olap::vector_t v1 {cub[0]+1.0F, cub[1],      cub[2]     };
    Olap::vector_t v2 {cub[0]+1.0F, cub[1]+1.0F, cub[2]     };
    Olap::vector_t v3 {cub[0],      cub[1]+1.0F, cub[2]     };
    Olap::vector_t v4 {cub[0],      cub[1],      cub[2]+1.0F};
    Olap::vector_t v5 {cub[0]+1.0F, cub[1],      cub[2]+1.0F};
    Olap::vector_t v6 {cub[0]+1.0F, cub[1]+1.0F, cub[2]+1.0F};
    Olap::vector_t v7 {cub[0],      cub[1]+1.0F, cub[2]+1.0F};

    Olap::Hexahedron hex {v0,v1,v2,v3,v4,v5,v6,v7};

    return Olap::overlap(sph, hex);
}//}}}

// caution : this function changes cub!
static inline float
sph_overlap (const float R,
             std::array<float,3> &cub,
             Interpolator<1024> *interp)
{//{{{
    // choose vertex closest to origin
    for (auto &x : cub)
        if (x < -0.5)
            x = - (x + 1.0F);

    //float expected = sph_overlap_exact(R, cub);

    float out;

    // check if cube completely outside sphere
    if (   ( (cub[0] < 0.0F) || (cub[1] < 0.0F) || (cub[2] < 0.0F) )
        && ( (cub[0] > R) || (cub[1] > R) || (cub[2] > R) ) )
        out = 0.0F;
    else if (   ( (cub[0] > 0.0F) && (cub[1] > 0.0F) && (cub[2] > 0.0F) )
             && ( std::hypot(cub[0], cub[1], cub[2]) > R ) )
        out = 0.0F;
    // check if sphere completely inside cube
    else if ( (cub[0] < -R) && (cub[1] < -R) && (cub[2] < -R) )
        out = 4.188790204786F * R * R * R;
    // check if we need to call the interpolator/exact calculation
    // TODO can first check the size of R
    else if (   (std::hypot(cub[0]+1.0F, cub[1]+1.0F, cub[2]+1.0F) > R)
             || (std::hypot(cub[0]+1.0F, cub[1]+1.0F, cub[2]     ) > R)
             || (std::hypot(cub[0]+1.0F, cub[1]     , cub[2]     ) > R)
             || (std::hypot(cub[0]     , cub[1]+1.0F, cub[2]+1.0F) > R)
             || (std::hypot(cub[0]     , cub[1]+1.0F, cub[2]     ) > R)
             || (std::hypot(cub[0]     , cub[1]     , cub[2]+1.0F) > R) )
    {
        if ( !interp || R < interp->Rmin || R > interp->Rmax )
            out = sph_overlap_exact(R, cub);
        else
        {
            // sort the coordinates
            std::sort(cub.begin(), cub.end());
            
            // no computation to do, indicate by sign that this case
            //     needs to be added to the network input batch
            out = -1.0F;
        }
    }
    // cube completely inside sphere
    else
        out = 1.0F;

    //std::cout << cub[0]/R << std::endl;
    //std::cout << cub[1]/R << std::endl;
    //std::cout << cub[2]/R << std::endl;
    //std::cout << expected << "\t" << out << std::endl;

    return out;
}//}}}

static inline float
line_overlap (const float x0, const float a0, const float x1, const float a1)
// assumes x1 > x0
{//{{{
    return std::max(0.0F, std::min(x0+a0-x1, a1));
}//}}}

static inline float
cub_overlap (const float part_half_side,
             const std::array<float,3> &cub)
{//{{{
    float out = 1.0F;
    for (int ii=0; ii<3; ii++)
    {
        float x0 = - part_half_side;
        float x1 = cub[ii];
        if (x1 > x0)
        {
            out *= line_overlap(x0, 2.0F*part_half_side, x1, 1.0F);
        }
        else
        {
            out *= line_overlap(x1, 1.0F, x0, 2.0F*part_half_side);
        }
    }
    return out;
}//}}}

static inline void
add_to_box (float *const __restrict__ box,
            const float *const __restrict__ field,
            const float vol, const long box_dim)
{//{{{
    // unroll the most common cases for efficiency
    switch (box_dim)
    {
        case (1) :
            #pragma omp atomic
            box[0] += field[0] * vol;
            break;
        case (2) :
            for (long dd = 0; dd < 2; ++dd)
                #pragma omp atomic
                box[dd] += field[dd] * vol;
            break;
        case (3) :
            for (long dd = 0; dd < 3; ++dd)
                #pragma omp atomic
                box[dd] += field[dd] * vol;
            break;
        default :
            for (long dd = 0; dd < box_dim; ++dd)
                #pragma omp atomic
                box[dd] += field[dd] * vol;
    }
}//}}}

int
voxelize (const long Nparticles, const long box_N, const float box_L, const long box_dim,
          const float *const __restrict__ coords,
          const float *const __restrict__  radii,
          const float *const __restrict__ field,
          float *const __restrict__ box,
          const wrk_mode mode, const char *const buf_str)
{//{{{
    const float box_a = box_L / (float)box_N;

    std::vector<Interpolator<1024>> interp;
    if (mode == interp_m)
        for (int ii = 0; ii != omp_get_max_threads(); ++ii)
            interp.push_back( Interpolator<1024>(std::string(buf_str)) );

    #pragma omp parallel for schedule(runtime)
    for (long pp = 0; pp < Nparticles; ++pp)
    {
        float R = radii[pp] / box_a;
        if (mode == cube_m) // cbrt(pi/6)
            R *= 0.80599597700823482F;

        std::array<float,3> part_centre;
        for (long ii = 0; ii < 3L; ++ii)
            part_centre[ii] = coords[3L*pp+ii] / box_a;
        
        for (long xx  = (long)(part_centre[0] - R) - 1;
                  xx <= (long)(part_centre[0] + R);
                ++xx)
        {
            const long idx_x = box_N * box_N * ((box_N+xx%box_N) % box_N);

            for (long yy  = (long)(part_centre[1] - R) - 1;
                      yy <= (long)(part_centre[1] + R);
                    ++yy)
            {
                const long idx_y = idx_x + box_N * ((box_N+yy%box_N) % box_N);

                __builtin_prefetch (box+idx_y+box_N, 1, 3);

                for (long zz  = (long)(part_centre[2] - R) - 1;
                          zz <= (long)(part_centre[2] + R);
                        ++zz)
                {
                    const long idx = idx_y + (box_N+zz%box_N) % box_N;
                    std::array<float,3> cub { (float)(xx) - part_centre[0],
                                              (float)(yy) - part_centre[1],
                                              (float)(zz) - part_centre[2] };

                    float vol = 0.0F; // to avoid maybe-uninitialized

                    switch (mode)
                    {
                        case (cube_m) :
                            vol = cub_overlap(R, cub);
                            break;
                        case (sphere_m) :
                            vol = sph_overlap(R, cub, nullptr);
                            break;
                        case (interp_m) :
                            vol = sph_overlap(R, cub, &interp[omp_get_thread_num()]);
                            break;
                        case (debug_m) :
                            vol = sph_overlap_exact(R, cub);
                            break;
                    }

                    if (vol == 0.0F)
                        continue;
                    else if (vol > 0.0F)
                        add_to_box(box+idx*box_dim, field+pp*box_dim, vol, box_dim);
                    else if (mode == interp_m)
                    {
                        // save typing
                        auto &tmp_interp = interp[omp_get_thread_num()];

                        tmp_interp.add_point(idx, pp, R, cub);

                        if ( tmp_interp.is_full() )
                        {
                            auto &res = tmp_interp.result();

                            for (int ii = 0; ii != tmp_interp.num_points; ++ii)
                                add_to_box(box+tmp_interp.idx[ii]*box_dim,
                                           field+tmp_interp.pidx[ii]*box_dim,
                                           res[ii][0], box_dim);

                            tmp_interp.reset();
                        }
                    }
                    else
                        throw std::runtime_error("should not be here");
                }
            }
        }
    }

    // clean up the remaining stuff stored in the interpolators
    if (mode == interp_m)
    {
        for (auto &tmp_interp : interp)
        {
            auto &res = tmp_interp.result();

            for (int ii = 0; ii != tmp_interp.num_points; ++ii)
                add_to_box(box+tmp_interp.idx[ii]*box_dim,
                           field+tmp_interp.pidx[ii]*box_dim,
                           res[ii][0], box_dim);

            tmp_interp.reset();
        }
    }

    return 0;
}//}}}
