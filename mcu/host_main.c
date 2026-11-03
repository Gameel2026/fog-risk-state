#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int run(const float *x, int n, float *out, uint8_t *g);
int main(int c, char **v){
  FILE *f=fopen(v[1],"rb"); int n; fread(&n,4,1,f);
  float *x=malloc(sizeof(float)*384*3*n); fread(x,4,384*3*n,f); fclose(f);
  float *o=malloc(sizeof(float)*17*n); uint8_t *g=malloc(n);
  run(x,n,o,g);
  f=fopen(v[2],"wb"); fwrite(o,4,17*n,f); fwrite(g,1,n,f); fclose(f); return 0;}
