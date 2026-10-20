/* gpu_kernels.cu: registry of the GPU ECM kernels (see gpu_kernels.h).

  Copyright 2026 Greg Childers

  This program is free software; you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation; either version 3 of the License, or (at your
  option) any later version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
  more details.

  You should have received a copy of the GNU General Public License along
  with this program; see the file COPYING.  If not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02111-1307, USA.
*/

#include <string.h>
#include "gpu_kernels.h"

void gpu_kset_00 (gpu_kernel_table *);
void gpu_kset_01 (gpu_kernel_table *);
void gpu_kset_02 (gpu_kernel_table *);
void gpu_kset_03 (gpu_kernel_table *);
void gpu_kset_04 (gpu_kernel_table *);
void gpu_kset_05 (gpu_kernel_table *);
void gpu_kset_06 (gpu_kernel_table *);
void gpu_kset_07 (gpu_kernel_table *);
void gpu_kset_08 (gpu_kernel_table *);
void gpu_kset_09 (gpu_kernel_table *);
void gpu_kset_10 (gpu_kernel_table *);
void gpu_kset_11 (gpu_kernel_table *);
void gpu_kset_12 (gpu_kernel_table *);
void gpu_kset_13 (gpu_kernel_table *);

const gpu_kernel_set *
gpu_kernels (uint32_t tpi, uint32_t L)
{
  static gpu_kernel_table t;
  static int init = 0;
  if (!init)
    {
      memset (&t, 0, sizeof (t));
      gpu_kset_00 (&t);
      gpu_kset_01 (&t);
      gpu_kset_02 (&t);
      gpu_kset_03 (&t);
      gpu_kset_04 (&t);
      gpu_kset_05 (&t);
      gpu_kset_06 (&t);
      gpu_kset_07 (&t);
      gpu_kset_08 (&t);
      gpu_kset_09 (&t);
      gpu_kset_10 (&t);
      gpu_kset_11 (&t);
      gpu_kset_12 (&t);
      gpu_kset_13 (&t);
      init = 1;
    }
  const gpu_kernel_set *k;
  if (tpi == 1)
    k = (L >= 1 && L <= GPU_SOLO_MAX_L) ? &t.solo[L] : NULL;
  else
    k = (L >= 1 && L <= GPU_DIST_MAX_L) ? &t.dist[L] : NULL;
  return (k != NULL && k->s1 != NULL) ? k : NULL;
}
