with cuda; use cuda;
with nvml_h;
with Interfaces.C;
with Ada.Integer_Text_IO;

procedure testml is
   device : nvml_h.nvmlDevice_t;
   cnt,temp : aliased Interfaces.C.unsigned;
begin
   checkNVML(nvml_h.nvmlInit_v2);
   checkNVML(nvml_h.nvmlDeviceGetCount_v2(cnt'Access));
   checkNVML(nvml_h.nvmlDeviceGetHandleByIndex_v2(0, device'Address));
   checkNVML(nvml_h.nvmlDeviceGetTemperature(device,nvml_h.NVML_TEMPERATURE_GPU,temp'Access));
   checkNVML(nvml_h.nvmlShutdown);
   Ada.Integer_Text_IO.Put(Integer(temp));
end testml;
