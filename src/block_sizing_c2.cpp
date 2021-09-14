#include "cake_c2.h"

int get_block_dim_C2(unsigned long long dev_dram_sz, double alpha_n, int p) {

   return (int) sqrt(((double) dev_dram_sz / (4*2*2.0)) / (2 + alpha_n*p));
}
