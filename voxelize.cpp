#include <iostream>

#include <omp.h>

#include "overlap_lft.hpp"

static inline float
sph_overlap (Olap::Sphere& sph, float cub[3])
{
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
}

static inline float
line_overlap (float x0, float a0, float x1, float a1)
// assumes x1 > x0
{
    return std::max(0.0F, std::min(x0+a0-x1, a1));
}

static inline float
cub_overlap (const float part_vert[3], float part_side, float cub[3])
{
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
}

int
voxelize (long Nparticles, long box_N, float box_L, long box_dim,
          const float *const coords, const float *const radii, const float *const field,
          float *box, int spherical)
{
    float box_a = box_L / (float)box_N;

    std::vector<Olap::Sphere> sph (omp_get_max_threads());

    #pragma omp parallel for schedule(runtime)
    for (long pp = 0; pp < Nparticles; ++pp)
    {
        float R = radii[pp] / box_a;
        if (!(spherical)) // cbrt(pi/6)
            R *= 0.8059959770082348203584834233196424694723070361619307778461460376F;

        float part_centre[3];
        for (long ii = 0; ii < 3L; ++ii)
            part_centre[ii] = coords[3L*pp+ii] / box_a;

        if (spherical)
            sph[omp_get_thread_num()] = Olap::Sphere {{part_centre[0],
                                                       part_centre[1],
                                                       part_centre[2]},
                                                      R};

        for (long xx  = (long)(part_centre[0] - R) - 1;
                  xx <= (long)(part_centre[0] + R);
                ++xx)
        {
            long idx_x = box_N * box_N * ((box_N+xx%box_N) % box_N);

            for (long yy  = (long)(part_centre[1] - R) - 1;
                      yy <= (long)(part_centre[1] + R);
                    ++yy)
            {
                long idx_y = idx_x + box_N * ((box_N+yy%box_N) % box_N);

                __builtin_prefetch (box+idx_y+box_N, 1, 3);

                for (long zz  = (long)(part_centre[2] - R) - 1;
                          zz <= (long)(part_centre[2] + R);
                        ++zz)
                {
                    long idx = idx_y + (box_N+zz%box_N) % box_N;

                    float cub[] = {(float)xx, (float)yy, (float)zz};
                    float vol;

                    if (spherical)
                        vol = sph_overlap(sph[omp_get_thread_num()], cub);
                    else
                        vol = cub_overlap(part_centre, 2.0F*R, cub);

                    // unroll the most common cases for efficiency
                    switch (box_dim)
                    {
                        case (1) :
                            #pragma omp atomic
                            box[idx] += field[pp] * vol;
                            break;
                        case (2) :
                            for (long dd = 0; dd < 2; ++dd)
                                #pragma omp atomic
                                box[idx*2 + dd] += field[pp*2 + dd] * vol;
                            break;
                        case (3) :
                            for (long dd = 0; dd < 3; ++dd)
                                #pragma omp atomic
                                box[idx*3 + dd] += field[pp*3 + dd] * vol;
                            break;
                        default :
                            for (long dd = 0; dd < box_dim; ++dd)
                                #pragma omp atomic
                                box[idx*box_dim + dd] += field[pp*box_dim + dd] * vol;
                    }
                }
            }
        }
    }

    return 0;
}
