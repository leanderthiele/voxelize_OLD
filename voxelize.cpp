#include <omp.h>

// TODO
#include <iostream>
#include <stdio.h>

#include "overlap_lft.hpp"

static inline float
overlap (Olap::Sphere& sph, float cub[3])
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

static inline long
flattened_index (long coord[3], long N)
{
    return N * N * ((N+coord[0]%N) % N)
           +   N * ((N+coord[1]%N) % N)
           +       ((N+coord[2]%N) % N);
}

int
voxelize (long Nparticles, long box_N, float box_L, long box_dim,
          const float *const coords, const float *const radii, const float *const field,
          float *box)
{
    float box_a = box_L / (float)box_N;

    #pragma omp parallel for
    for (long pp = 0; pp < Nparticles; ++pp)
    {
        float R = radii[pp] / box_a;

        float sphere_centre[3];
        for (long ii = 0; ii < 3L; ++ii)
            sphere_centre[ii] = coords[3L*pp+ii] / box_a;

        Olap::Sphere sph {{sphere_centre[0], sphere_centre[1], sphere_centre[2]}, R};

        for (long xx  = (long)(sphere_centre[0] - R) - 1;
                  xx <= (long)(sphere_centre[0] + R);
                ++xx)
        {
            for (long yy  = (long)(sphere_centre[1] - R) - 1;
                      yy <= (long)(sphere_centre[1] + R);
                    ++yy)
            {
                for (long zz  = (long)(sphere_centre[2] - R) - 1;
                          zz <= (long)(sphere_centre[2] + R);
                        ++zz)
                {
                    float cub[] = {(float)xx, (float)yy, (float)zz};
                    float vol = overlap(sph, cub);

                    long pos[] = {xx, yy, zz};
                    long idx = flattened_index(pos, box_N);

                    // unroll the most common cases for efficiency
                    if (box_dim == 1)
                    {
                        #pragma omp atomic
                        box[idx] += field[pp] * vol;
                    }
                    else if (box_dim == 2)
                    {
                        for (long dd = 0; dd < 2; ++dd)
                        {
                            #pragma omp atomic
                            box[idx*2 + dd] += field[pp*2 + dd] * vol;
                        }
                    }
                    else if (box_dim == 3)
                    {
                        for (long dd = 0; dd < 3; ++dd)
                        {
                            #pragma omp atomic
                            box[idx*3 + dd] += field[pp*3 + dd] * vol;
                        }
                    }
                    else
                    {
                        for (long dd = 0; dd < box_dim; ++dd)
                        {
                            #pragma omp atomic
                            box[idx*box_dim + dd] += field[pp*box_dim + dd] * vol;
                        }
                    }
                }
            }
        }
    }

    return 0;
}
