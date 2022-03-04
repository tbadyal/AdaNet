with cuda_h;
with nvrtc_h;
with Interfaces.c.Strings;
with driver_types_h;
with cuda_runtime_api_h;
with cudnn_h;
with cublas_h;
with cublas_api_h;

package body cuda is

   procedure checkCUDA(status : cuda_h.CUresult)
   is
      use type cuda_h.CUresult;
   begin
      if status /= cuda_h.CUDA_SUCCESS then
         declare
            errname, errstr : Interfaces.c.Strings.chars_ptr;
            lstatus1 : cuda_h.CUresult := cuda_h.cuGetErrorname(status,errname'address);
            lstatus2 : cuda_h.CUresult := cuda_h.cuGetErrorString(status,errstr'address);
         begin
            if lstatus1 /= cuda_h.CUDA_SUCCESS or lstatus2 /= cuda_h.CUDA_SUCCESS then
               raise Program_Error with "UNKNOWN CUDA ERROR: " & lstatus1'Image;
            else
               raise Program_Error with "CUDA ERROR: " &  Interfaces.c.Strings.Value(errname) & " : " & Interfaces.c.Strings.Value(errstr);
            end if;
         end;
      end if;
   end checkCUDA;

   procedure checkNVRTC(status : nvrtc_h.nvrtcResult)
   is
      use type nvrtc_h.nvrtcResult;
   begin
      if  status = nvrtc_h.NVRTC_ERROR_COMPILATION then
         raise NVRTC_ERROR_COMPILATION;
      elsif status /= nvrtc_h.NVRTC_SUCCESS then
         raise Program_Error with "NVRTC ERROR: " & Interfaces.c.Strings.Value(nvrtc_h.nvrtcGetErrorString(status));
      end if;
   end checkNVRTC;

   procedure checkCUDART (status : driver_types_h.cudaError_t) is
      use type driver_types_h.cudaError_t;
   begin
      if status /= driver_types_h.cudaSuccess then
         raise Program_Error with "CUDART ERROR: " &  Interfaces.c.Strings.Value(cuda_runtime_api_h.cudaGetErrorName(status)) &
         " : " & Interfaces.c.Strings.Value(cuda_runtime_api_h.cudaGetErrorString(status));
      end if;

   end checkCUDART;

   procedure checkCUDNN(status : cudnn_h.cudnnStatus_t) is
      use type cudnn_h.cudnnStatus_t;
   begin
      if status /= cudnn_h.CUDNN_STATUS_SUCCESS then
         raise Program_Error with "CUDNN ERROR: " & Interfaces.c.Strings.Value(cudnn_h.cudnnGetErrorString(status));
       end if;
   end checkCUDNN;

   procedure checkCUBLAS(status : cublas_api_h.cublasStatus_t) is
      use type cublas_api_h.cublasStatus_t;
   begin
      if status /= cublas_api_h.CUBLAS_STATUS_SUCCESS then
         raise Program_Error with "CUBLAS ERROR: " & integer'image(integer(cublas_h.cublasGetError));
      end if;
   end checkCUBLAS;

   procedure checkCURAND(status : curand_h.curandStatus_t) is
      use type curand_h.curandStatus_t;
   begin
      if status /= curand_h.CURAND_STATUS_SUCCESS then
         raise Program_Error with "CURAND ERROR: " & integer'image(integer(status));
      end if;
   end checkCURAND;

   procedure checkNVML(status : nvml_h.nvmlReturn_t) is
      use type nvml_h.nvmlReturn_t;
   begin
      if status /= nvml_h.NVML_SUCCESS then
         raise Program_Error with "NVML ERROR: " & Interfaces.C.Strings.Value(nvml_h.nvmlErrorString(status));
      end if;
   end checkNVML;

end cuda;
