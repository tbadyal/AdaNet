with Interfaces.C.Strings;
with stddef_h;
with ada.Exceptions;
with nvrtc_h;
with System;

package body cuda.driver is

   function init(name, str : String) return kernel is
      program : aliased nvrtc_h.nvrtcProgram;
   begin
      return self : kernel do
         self.name := Ada.Strings.Unbounded.To_Unbounded_String(name);
         self.str := ada.Strings.Unbounded.To_Unbounded_String(str);

         checkNVRTC(nvrtc_h.nvrtcCreateProgram(prog         => program'Address,
                                               src          => Interfaces.c.Strings.New_String(ada.Strings.Unbounded.To_String(self.str)),
                                               name         => Interfaces.c.Strings.New_String(ada.Strings.Unbounded.To_String(self.name)),
                                               numHeaders   => 0,
                                               headers      => System.Null_Address,
                                               includeNames => System.Null_Address));

         checkNVRTC(nvrtc_h.nvrtcCompileProgram(prog       => program,
                                                numOptions => Interfaces.c.int(opt'Length),
                                                options    => opt'Address));

         checkNVRTC(nvrtc_h.nvrtcGetPTXSize(prog       => program,
                                            ptxSizeRet => self.ptx_size'Access));

         declare
            ptx_str : String(1..Integer(self.ptx_size));
            pragma Warnings(off, ptx_str);
         begin
            self.ptx := Interfaces.c.Strings.New_String(ptx_str);
         end;

         checkNVRTC(nvrtc_h.nvrtcGetPTX(prog => program,
                                        ptx  => self.ptx));

         map.Insert(self.name, self);

         checkNVRTC(nvrtc_h.nvrtcDestroyProgram(program'Address));

      end return;

   exception
      when e : NVRTC_ERROR_COMPILATION =>

         declare
            log : Interfaces.c.Strings.chars_ptr;
            log_size : aliased stddef_h.size_t;
         begin

            checkNVRTC(nvrtc_h.nvrtcGetProgramLogSize(prog       => program,
                                                      logSizeRet => log_size'Access));
            declare
               log_str : String(1..Integer(log_size));
               pragma Warnings(off, log_str);
            begin
               log := Interfaces.c.Strings.New_String(log_str);
            end;

            checkNVRTC(nvrtc_h.nvrtcGetProgramLog(prog => program,
                                                  log  => log));

            ada.Exceptions.Raise_Exception(ada.Exceptions.Exception_Identity(e),Interfaces.c.Strings.Value(log));
         end;

   end init;




   procedure compile is

      linker : cuda_h.CUlinkState;
      module : cuda_h.CUmodule;
      cubin_size : aliased stddef_h.size_t;
      cubin : Interfaces.c.Strings.char_array_access;

   begin


      checkCUDA(cuda_h.cuLinkCreate_v2(numOptions   => 0,
                                       options      => null,
                                       optionValues => System.Null_Address,
                                       stateOut     => linker'Address));

      for i in map.Iterate loop

         declare
            ptx_str : Interfaces.c.char_array :=Interfaces.c.Strings.value(map(i).ptx);
         begin

            checkCUDA(cuda_h.cuLinkAddData_v2(state        => linker,
                                              c_type       => cuda_h.CU_JIT_INPUT_PTX,
                                              data         => ptx_str'Address,
                                              size         => map(i).ptx_size,
                                              name         => Interfaces.c.Strings.New_String(ada.Strings.Unbounded.To_String(map(i).name)),
                                              numOptions   => 0,
                                              options      => null,
                                              optionValues => System.Null_Address));

         end;


      end loop;


      checkCUDA(cuda_h.cuLinkComplete(state    => linker,
                                      cubinOut => cubin'Address,
                                      sizeOut  => cubin_size'Access));

      checkCUDA(cuda_h.cuModuleLoadData(module => module'Address,
                                        image => cubin.all'address));




      for i in map.Iterate loop

         checkCUDA(cuda_h.cuModuleGetFunction(hfunc => map(i).func'Address,
                                              hmod => module,
                                              name => Interfaces.c.Strings.New_String(ada.Strings.Unbounded.To_String(map(i).name))));
      end loop;



      checkCUDA(cuda_h.cuLinkDestroy(state => linker));

   end compile;

   procedure exec(self : kernel; num : Integer; args : arguments) is
   begin

      checkCUDA(cuda_h.cuLaunchKernel(f  => map(self.name).func,
                                      gridDimX  => Interfaces.c.unsigned(Float'Ceiling(float(num)/1024.0)),
                                      gridDimY  => Interfaces.c.unsigned(1),
                                      gridDimZ  => Interfaces.c.unsigned(1),
                                      blockDimX  => Interfaces.c.unsigned(1024),
                                      blockDimY  => Interfaces.c.unsigned(1),
                                      blockDimZ  => Interfaces.c.unsigned(1),
                                      sharedMemBytes  => Interfaces.c.unsigned(0),
                                      hStream  => cuda_h.CUstream (System.Null_Address),
                                      kernelParams => args'Address,
                                      extra => System.Null_Address));
   end exec;

   procedure exec(self : kernel; blocks,threads : Integer; args : arguments) is
   begin

      checkCUDA(cuda_h.cuLaunchKernel(f  => map(self.name).func,
                                      gridDimX  => Interfaces.c.unsigned(blocks),
                                      gridDimY  => Interfaces.c.unsigned(1),
                                      gridDimZ  => Interfaces.c.unsigned(1),
                                      blockDimX  => Interfaces.c.unsigned(threads),
                                      blockDimY  => Interfaces.c.unsigned(1),
                                      blockDimZ  => Interfaces.c.unsigned(1),
                                      sharedMemBytes  => Interfaces.c.unsigned(0),
                                      hStream  => cuda_h.CUstream (System.Null_Address),
                                      kernelParams => args'Address,
                                      extra => System.Null_Address));
   end exec;



begin

   checkCUDA(cuda_h.cuInit(Flags => 0));

   checkCUDA(cuda_h.cuDeviceGet(device => device'Access,
                                ordinal => 0));

   checkCUDA(cuda_h.cuCtxCreate_v2(pctx => context'Address,
                                   flags => 0,
                                   dev => device));

end cuda.driver;
