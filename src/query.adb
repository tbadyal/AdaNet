with cuda;
with Ada.Text_IO;
with cuda_runtime_api_h;
with Interfaces.c;
with driver_types_h;
with ada.Strings.Fixed;

procedure query is
   device_cnt, device_id : aliased Interfaces.c.int;
   device_prop : aliased driver_types_h.cudaDeviceProp;
begin
   cuda.checkCUDART(cuda_runtime_api_h.cudaGetDeviceCount(count => device_cnt'Access));
   Ada.Text_IO.Put_Line("Device Count:" & device_cnt'Img);
   cuda.checkCUDART(cuda_runtime_api_h.cudaGetDevice(device => device_id'Access));
   Ada.Text_IO.Put_Line("Device ID:" & device_id'Img);
   cuda.checkCUDART(cuda_runtime_api_h.cudaGetDeviceProperties(prop   => device_prop'Access,
                                                               device => device_id));
   ada.Text_IO.Put_Line("Device Name:" & Interfaces.c.To_Ada(device_prop.name));
   ada.Text_IO.Put_Line("Compute Compatibility:" & device_prop.major'img & "." &
                          ada.Strings.Fixed.Trim(device_prop.minor'img, ada.Strings.Both));
   Ada.Text_IO.Put_Line("Threads Per Block:" & device_prop.maxThreadsPerBlock'img);
end query;
