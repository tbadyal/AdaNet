with interfaces.c;

package body cuda.io is

   function malloc(size : stddef_h.size_t) return System.Address is
   begin
      return dx : System.Address do
         cuda.checkCUDART(cuda_runtime_api_h.cudaMalloc(dx'Address, size));
      end return;
   end malloc;


   procedure free(dx : System.Address) is
   begin
      cuda.checkCUDART(cuda_runtime_api_h.cudaFree(dx));
   end free;

   procedure to_device(x : System.Address; dx : System.Address; size : stddef_h.size_t) is
   begin
         cuda.checkCUDART(cuda_runtime_api_h.cudaMemcpy(dx, x, size, driver_types_h.cudaMemcpyHostToDevice));
   end to_device;

   procedure to_host(dx : System.Address; x : System.Address; size : stddef_h.size_t) is
   begin
      cuda.checkCUDART(cuda_runtime_api_h.cudaMemcpy(x, dx, size, driver_types_h.cudaMemcpyDeviceToHost));
   end to_host;

   procedure memset(dx : System.Address; size : stddef_h.size_t; val : Integer) is
   begin
      cuda.checkCUDART(cuda_runtime_api_h.cudaMemset(dx, interfaces.c.int(val), size));
   end memset;

   procedure memcopy(dx, dy : System.Address; size : stddef_h.size_t) is
   begin
        cuda.checkCUDART(cuda_runtime_api_h.cudaMemcpy(dy, dx, size, driver_types_h.cudaMemcpyDeviceToDevice));
   end memcopy;

   function mallocManaged(size : stddef_h.size_t) return System.Address is
   begin
      return dx : System.Address do
         cuda.checkCUDART(cuda_runtime_api_h.cudaMallocManaged(dx'Address, size, 2));
      end return;
   end mallocManaged;



end cuda.io;
