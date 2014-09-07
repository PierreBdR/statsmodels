#include "grid_interp.h"

intptr_t binary_search(double value,
                       double *mesh,
                       intptr_t inf, intptr_t sup)
{
  uintptr_t cur;
  while(sup > inf)
  {
    cur = inf + ((sup-inf)>>1);
    if(value >= mesh[cur])
      inf = cur + 1;
    else
      sup = cur;
  }
  return inf - 1;
}
