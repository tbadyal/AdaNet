with Interfaces.C;
with Ada.Text_IO;

package body cuda.util is

   procedure createhandles is
   begin
      checkNVML(nvml_h.nvmlInit_v2);
      checkNVML(nvml_h.nvmlDeviceGetHandleByIndex_v2(0, device'Address));
   end createhandles;

   procedure destroyhandles is
   begin
      checkNVML(nvml_h.nvmlShutdown);
   end destroyhandles;

   procedure setdevice(id : Integer := 0) is
   begin
      checkCUDART(cuda_runtime_api_h.cudaSetDevice(Interfaces.C.int(id)));
   end setdevice;

   function get_gpu_temp return integer is
      temp : aliased Interfaces.C.unsigned;
   begin
      checkNVML(nvml_h.nvmlDeviceGetTemperature(device,nvml_h.NVML_TEMPERATURE_GPU,temp'Access));
      return Integer(temp);
   end get_gpu_temp;

begin
   createhandles;
end cuda.util;
