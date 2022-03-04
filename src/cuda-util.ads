
package cuda.util is

   device : nvml_h.nvmlDevice_t;

   procedure createhandles;
   procedure destroyhandles;
   procedure setdevice(id : Integer := 0);
   function get_gpu_temp return Integer;
end cuda.util;
