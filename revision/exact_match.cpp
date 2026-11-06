#include <vector>
#include <limits>
#include <cstdint>
#include <cstring>
#include <intrin.h>
#include <immintrin.h>
extern "C" __declspec(dllexport) int match_exact(const void* aa,const void* bb,int n,int m,int dim,int type,int mutual,int* oi,int* oj) {
 std::vector<double> ar(n,1e100),as(n,1e100),br(m,1e100),bs(m,1e100);
 std::vector<int> ai(n,-1),bi(m,-1);
 for(int i=0;i<n;i++) for(int j=0;j<m;j++) {
  double d=0;
  if(type==0) {
   const uint8_t* a=(const uint8_t*)aa+i*dim;const uint8_t* b=(const uint8_t*)bb+j*dim;
   int k=0;for(;k+8<=dim;k+=8) {uint64_t x,y;memcpy(&x,a+k,8);memcpy(&y,b+k,8);d+=__popcnt64(x^y);}
   for(;k<dim;k++)d+=__popcnt((unsigned)(a[k]^b[k]));
  } else {
   const float* a=(const float*)aa+i*dim;const float* b=(const float*)bb+j*dim;
   __m256 sums=_mm256_setzero_ps();int k=0;
   for(;k+8<=dim;k+=8){__m256 v=_mm256_sub_ps(_mm256_loadu_ps(a+k),_mm256_loadu_ps(b+k));sums=_mm256_add_ps(sums,_mm256_mul_ps(v,v));}
   alignas(32) float vals[8];_mm256_store_ps(vals,sums);float accum=0;for(int z=0;z<8;z++)accum+=vals[z];
   for(;k<dim;k++){float v=a[k]-b[k];accum+=v*v;}d=accum;
  }
  if(d<ar[i]){as[i]=ar[i];ar[i]=d;ai[i]=j;}else if(d<as[i])as[i]=d;
  if(d<br[j]){bs[j]=br[j];br[j]=d;bi[j]=i;}else if(d<bs[j])bs[j]=d;
 }
 int out=0;double ratio=type==0?.75:.5625;
 for(int i=0;i<n;i++){int j=ai[i];if(j>=0&&ar[i]<ratio*as[i]&&(!mutual||(bi[j]==i&&br[j]<ratio*bs[j]))){oi[out]=i;oj[out]=j;out++;}}
 return out;
}
