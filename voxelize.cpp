#include <iostream>
#include <array>
#include <algorithm>

#include <omp.h>

#include "overlap_lft.hpp"
#include "_interpolate.hpp"

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

static inline float
sph_overlap (const std::array<float,3> &part_centre,
             float R,
             std::array<float,3> &cub,
             Interpolator<4,float,float>  &interp)
{
    // subtract the origin
    for (int ii=0; ii != 3; ii++)
        cub[ii] -= part_centre[ii];

    // TODO debugging
    std::cout << "exact : " << sph_overlap_exact(R, cub) << std::endl;

    // choose vertex closest to origin
    for (auto &x : cub)
        if (x < -0.5)
            x = - (x + 1.0F);

    float out;

    // check if cube completely outside sphere
    if (std::hypot(cub[0], cub[1], cub[2]) > R)
        out = 0.0F;
    // check if sphere completely inside cube
    else if ( (cub[0] < -R) && (cub[1] < -R) && (cub[2] < -R) )
        out =  4.188790204786F * R * R * R;
    // check if we need to call the interpolator
    else if (   (std::hypot(cub[0]+1.0F, cub[1]+1.0F, cub[2]+1.0F) > R)
             || (std::hypot(cub[0]+1.0F, cub[1]+1.0F, cub[2]     ) > R)
             || (std::hypot(cub[0]+1.0F, cub[1]     , cub[2]     ) > R)
             || (std::hypot(cub[0]     , cub[1]+1.0F, cub[2]+1.0F) > R)
             || (std::hypot(cub[0]     , cub[1]+1.0F, cub[2]     ) > R)
             || (std::hypot(cub[0]     , cub[1]     , cub[2]+1.0F) > R) )
    {
        // sort the coordinates
        std::sort(cub.begin(), cub.end());

        // write argument for the interpolator
        float x[4] = { R, cub[0], cub[1], cub[2] };
        try
        {
            out = interp.eval(x);
            std::cout << "used interpolator" << std::endl;
        }
        catch (const InterpolatorExcept &e)
        {
            // TODO debugging
            std::cout << "interpolator did not work" << std::endl;
            if (e.level == 4) // expected case, radius not interpolated
                out = sph_overlap_exact(R, cub);
            else // unexpected case
            {
                std::cout << "level = " << e.level << std::endl;
                std::cout << "R = " << R << "\tcub = ";
                for (auto &x : cub)
                    std::cout << " " << x << " ";
                std::cout << std::endl;
                std::cout << e.what() << std::endl;
                throw;
            }
        }
    }
    // cube completely inside sphere
    else
        out = 1.0F;

    // TODO debugging
    std::cout << "computed : " << out << "\n" << std::endl;

    return out;
}


static inline float
line_overlap (const float x0, const float a0, const float x1, const float a1)
// assumes x1 > x0
{//{{{
    return std::max(0.0F, std::min(x0+a0-x1, a1));
}//}}}

static inline float
cub_overlap (const std::array<float,3> &part_vert,
             const float part_side,
             const std::array<float,3> &cub)
{//{{{
    float out = 1.0F;
    for (int ii=0; ii<3; ii++)
    {
        float x0 = part_vert[ii] - 0.5F * part_side;
        float x1 = cub[ii];
        if (x1 > x0)
        {
            out *= line_overlap(x0, part_side, x1, 1.0F);
        }
        else
        {
            out *= line_overlap(x1, 1.0F, x0, part_side);
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
          const int spherical)
{//{{{
    float box_a = box_L / (float)box_N;

    Interpolator<4,float,float> interp;

    if (spherical)
        interp.load("test.bin");

    #pragma omp parallel for schedule(runtime)
    for (long pp = 0; pp < Nparticles; ++pp)
    {
        float R = radii[pp] / box_a;
        if (!(spherical)) // cbrt(pi/6)
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

                    std::array<float,3> cub = {(float)xx, (float)yy, (float)zz};
                    float vol;

                    if (spherical)
                        vol = sph_overlap(part_centre, R, cub, interp);
                    else
                        vol = cub_overlap(part_centre, 2.0F*R, cub);

                    add_to_box(box+idx*box_dim, field+pp*box_dim, vol, box_dim);
                }
            }
        }
    }

    return 0;
}//}}}
