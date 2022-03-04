with cuda_h;
with nvrtc_h;
with driver_types_h;
with cudnn_h;
with cublas_api_h;
with curand_h;
with cuda_runtime_api_h;
with stddef_h;
with nvml_h;

package cuda is

   NVRTC_ERROR_COMPILATION : exception;

   procedure checkCUDA(status : cuda_h.CUresult);
   procedure checkNVRTC(status : nvrtc_h.nvrtcResult);
   procedure checkCUDART(status : driver_types_h.cudaError_t);
   procedure checkCUDNN(status : cudnn_h.cudnnStatus_t);
   procedure checkCUBLAS(status : cublas_api_h.cublasStatus_t) ;
   procedure checkCURAND(status : curand_h.curandStatus_t);
   procedure checkNVML(status : nvml_h.nvmlReturn_t);

end cuda;
