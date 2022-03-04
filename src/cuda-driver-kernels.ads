
package cuda.driver.kernels is

   tensor_add : kernel := init("__add",
                               "extern ""C"" __global__ "&
                                 " void __add(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] + y[tid];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_sub : kernel := init("__sub",
                               "extern ""C"" __global__ "&
                                 " void __sub(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] - y[tid];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_mul : kernel := init("__mul",
                               "extern ""C"" __global__ "&
                                 " void __mul(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] * y[tid];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_div : kernel := init("__div",
                               "extern ""C"" __global__ "&
                                 " void __div(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] / y[tid];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_pow : kernel := init("__pow",
                               "extern ""C"" __global__ "&
                                 " void __pow(int n, float *x, int y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = pow(x[tid],y);"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_sqrt : kernel := init("__sqrt",
                                "extern ""C"" __global__ "&
                                  " void __sqrt(int n, float *x) {"&
                                  "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                  "    if (tid < n) {"&
                                  "        x[tid] = sqrt(x[tid]);"&
                                  "    }"&
                                  " __syncthreads();"&
                                  "  }");

   tensor_scal_add : kernel := init("__scal_add",
                                    "extern ""C"" __global__ "&
                                      " void __scal_add(int n, float *x, float y) {"&
                                      "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                      "    if (tid < n) {"&
                                      "        x[tid] = x[tid] + y;"&
                                      "    }"&
                                      " __syncthreads();"&
                                      "  }");

   tensor_scal_sub : kernel := init("__scal_sub",
                                    "extern ""C"" __global__ "&
                                      " void __scal_sub(int n, float *x, float y) {"&
                                      "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                      "    if (tid < n) {"&
                                      "        x[tid] = x[tid] - y;"&
                                      "    }"&
                                      " __syncthreads();"&
                                      "  }");

   tensor_scal_mul : kernel := init("__scal_mul",
                                 "extern ""C"" __global__ "&
                                   " void __scal_mul(int n, float *x, float y) {"&
                                   "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                   "    if (tid < n) {"&
                                   "        x[tid] = x[tid] * y;"&
                                   "    }"&
                                   " __syncthreads();"&
                                   "  }");

   tensor_scal_div : kernel := init("__scal_div",
                                 "extern ""C"" __global__ "&
                                   " void __scal_div(int n, float *x, float y) {"&
                                   "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                   "    if (tid < n) {"&
                                   "        x[tid] = x[tid] / y;"&
                                   "    }"&
                                  " __syncthreads();"&
                                      "  }");

   adam_kernel : kernel := init("__adam",
                                "extern ""C"" __global__ "&
                                  " void __adam(int n, float *f, float *df, float *vdf, float *sdf, "&
                                  " int t, float alpha, float beta1, float beta2, float epsilon, float wd) {"&
                                  "  float vdf_c;"&
                                  "  float sdf_c;"&
                                  "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                  "    if (tid < n) {"&
                                  "        vdf[tid] = beta1 * vdf[tid] + (1.0 - beta1) *  0.5*df[tid];"&
                                  "        sdf[tid] = beta2 * sdf[tid] + (1.0 - beta2) *  0.5*pow(df[tid],2);"&
                                  "        vdf_c = vdf[tid] / (1.0 - pow(beta1, t));"&
                                  "        sdf_c = sdf[tid] / (1.0 - pow(beta2, t));"&
                                  "        f[tid] = f[tid] - alpha * (vdf_c / (sqrt(sdf_c) + epsilon)) - (wd * f[tid]);"&
                                  "    }"&
                                  "    __syncthreads(); "&
                                  "  }");

   calc_diff : kernel := init("__calc_diff",
                               "extern ""C"" __global__ "&
                                 " void __calc_diff(int n, float *x, float *dy, float *dx) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "       if (x[tid] == 0) {"&
                                 "          x[tid] == 1.0e-8;"&
                                 "        }"&
                                 "        else if (x[tid] == 1) {"&
                                 "            x[tid] == 1-1.0e-8;"&
                                 "          }"&
                                 "        dx[tid] = 1 - dy[tid] / x[tid];"&
                                 "    }"&
                                 " __syncthreads();"&
                                "  }");

   calc_loss : kernel := init("__calc_loss",
                               "extern ""C"" __global__ "&
                                 " void __calc_loss(int n, float *x, float *dy, float *y) {"&
                                "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                "    if (tid < n) {"&
                                "       if (dy[tid] == 1) {"&
                                "         if (x[tid] == 0) {"&
                                "            x[tid] == 1.0e-8;"&
                                "          }"&
                                "          else if (x[tid] == 1) {"&
                                "            x[tid] == 1-1.0e-8;"&
                                "          }"&
                                "          y[tid] = -log(x[tid]);"&
                                "       }"&
                                "       else {"&
                                "              y[tid] = 0;"&
                                "       }"&
                                "    }"&
                                " __syncthreads();"&
                                "  }");

   tensor_copy : kernel := init("__copy",
                                "extern ""C"" __global__ "&
                                  " void __copy(int n, float *x, float *y) {"&
                                  "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                  "    if (tid < n) {"&
                                  "        x[tid] = y[tid];"&
                                  "    }"&
                                  " __syncthreads();"&
                                  "  }");

   tensor_add_ex : kernel := init("__add_ex",
                               "extern ""C"" __global__ "&
                                 " void __add_ex(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] + y[tid - blockDim.x * blockIdx.x];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_sub_ex : kernel := init("__sub_ex",
                               "extern ""C"" __global__ "&
                                 " void __sub_ex(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] - y[tid - blockDim.x * blockIdx.x];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_mul_ex : kernel := init("__mul_ex",
                               "extern ""C"" __global__ "&
                                 " void __mul_ex(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] * y[tid - blockDim.x * blockIdx.x];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_div_ex : kernel := init("__div_ex",
                               "extern ""C"" __global__ "&
                                 " void __div_ex(int n, float *x, float *y) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        x[tid] = x[tid] / y[tid - blockDim.x * blockIdx.x];"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_add_scal_ex : kernel := init("__add_scal_ex",
                                       "extern ""C"" __device__ struct int4d { int n; int c; int h; int w;};"&
                                         "extern ""C"" __device__ "&
                                         "int4d _1d_to_4d(int idx, int ln, int lc, int lh, int lw)"&
                                         "{ int4d vec;"&
                                         "vec.w = idx % lw;"&
                                         "vec.h = ((idx - vec.w) / lw) % lh;"&
                                         "vec.c = ((idx - vec.h * lw - vec.w) / (lw * lh)) % lc;"&
                                         "vec.n = ((idx - vec.c * lh * lw - vec.h * lw - vec.w) / (lw * lh * lc)) % ln;"&
                                         "return vec;"&
                                         "};"&
                               "extern ""C"" __global__ "&
                                 " void __add_scal_ex(int n, float *x, float r, float g, float b, int ln, int lc, int lh, int lw) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        switch(_1d_to_4d(tid,ln,lc,lh,lw).c){"&
                                 "          case 0:"&
                                 "             x[tid] = x[tid] + r;"&
                                 "             break;"&
                                 "          case 1:"&
                                 "             x[tid] = x[tid] + g;"&
                                 "             break;"&
                                 "          case 2:"&
                                 "             x[tid] = x[tid] + b;"&
                                 "             break;"&
                                 "         }"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

  tensor_sub_scal_ex : kernel := init("__sub_scal_ex",
                                       "extern ""C"" __device__ struct int4d { int n; int c; int h; int w;};"&
                                         "extern ""C"" __device__ "&
                                         "int4d _1d_to_4d(int idx, int ln, int lc, int lh, int lw)"&
                                         "{ int4d vec;"&
                                         "vec.w = idx % lw;"&
                                         "vec.h = ((idx - vec.w) / lw) % lh;"&
                                         "vec.c = ((idx - vec.h * lw - vec.w) / (lw * lh)) % lc;"&
                                         "vec.n = ((idx - vec.c * lh * lw - vec.h * lw - vec.w) / (lw * lh * lc)) % ln;"&
                                         "return vec;"&
                                         "};"&
                               "extern ""C"" __global__ "&
                                 " void __sub_scal_ex(int n, float *x, float r, float g, float b, int ln, int lc, int lh, int lw) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        switch(_1d_to_4d(tid,ln,lc,lh,lw).c){"&
                                 "          case 0:"&
                                 "             x[tid] = x[tid] - r;"&
                                 "             break;"&
                                 "          case 1:"&
                                 "             x[tid] = x[tid] - g;"&
                                 "             break;"&
                                 "          case 2:"&
                                 "             x[tid] = x[tid] - b;"&
                                 "             break;"&
                                 "         }"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_mul_scal_ex : kernel := init("__mul_scal_ex",
                                       "extern ""C"" __device__ struct int4d { int n; int c; int h; int w;};"&
                                         "extern ""C"" __device__ "&
                                         "int4d _1d_to_4d(int idx, int ln, int lc, int lh, int lw)"&
                                         "{ int4d vec;"&
                                         "vec.w = idx % lw;"&
                                         "vec.h = ((idx - vec.w) / lw) % lh;"&
                                         "vec.c = ((idx - vec.h * lw - vec.w) / (lw * lh)) % lc;"&
                                         "vec.n = ((idx - vec.c * lh * lw - vec.h * lw - vec.w) / (lw * lh * lc)) % ln;"&
                                         "return vec;"&
                                         "};"&
                               "extern ""C"" __global__ "&
                                 " void __mul_scal_ex(int n, float *x, float r, float g, float b, int ln, int lc, int lh, int lw) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        switch(_1d_to_4d(tid,ln,lc,lh,lw).c){"&
                                 "          case 0:"&
                                 "             x[tid] = x[tid] * r;"&
                                 "             break;"&
                                 "          case 1:"&
                                 "             x[tid] = x[tid] * g;"&
                                 "             break;"&
                                 "          case 2:"&
                                 "             x[tid] = x[tid] * b;"&
                                 "             break;"&
                                 "         }"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   tensor_div_scal_ex : kernel := init("__div_scal_ex",
                                       "extern ""C"" __device__ struct int4d { int n; int c; int h; int w;};"&
                                         "extern ""C"" __device__ "&
                                         "int4d _1d_to_4d(int idx, int ln, int lc, int lh, int lw)"&
                                         "{ int4d vec;"&
                                         "vec.w = idx % lw;"&
                                         "vec.h = ((idx - vec.w) / lw) % lh;"&
                                         "vec.c = ((idx - vec.h * lw - vec.w) / (lw * lh)) % lc;"&
                                         "vec.n = ((idx - vec.c * lh * lw - vec.h * lw - vec.w) / (lw * lh * lc)) % ln;"&
                                         "return vec;"&
                                         "};"&
                               "extern ""C"" __global__ "&
                                 " void __div_scal_ex(int n, float *x, float r, float g, float b, int ln, int lc, int lh, int lw) {"&
                                 "    int tid = blockDim.x * blockIdx.x + threadIdx.x;"&
                                 "    if (tid < n) {"&
                                 "        switch(_1d_to_4d(tid,ln,lc,lh,lw).c){"&
                                 "          case 0:"&
                                 "             x[tid] = x[tid] / r;"&
                                 "             break;"&
                                 "          case 1:"&
                                 "             x[tid] = x[tid] / g;"&
                                 "             break;"&
                                 "          case 2:"&
                                 "             x[tid] = x[tid] / b;"&
                                 "             break;"&
                                 "         }"&
                                 "    }"&
                                 " __syncthreads();"&
                                 "  }");

   procedure autoInit;

end cuda.driver.kernels;
