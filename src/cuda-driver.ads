with ada.Strings.Unbounded.Hash;
with Interfaces.c.Strings;
with ada.Containers.Indefinite_Hashed_Maps;
with System;
with stddef_h;

package cuda.driver is

   use type Ada.Strings.Unbounded.Unbounded_String;

   device : aliased Interfaces.C.int;
   context : cuda_h.CUcontext;

   opt :  Interfaces.c.Strings.chars_ptr_array :=  (Interfaces.c.Strings.New_String("--gpu-architecture=compute_61"),
                                                    Interfaces.c.Strings.New_String("--fmad=false"),
                                                    Interfaces.c.Strings.New_String("--include-path=/usr/local/cuda/include"),
                                                    Interfaces.c.Strings.New_String("--include-path=/usr/include/x86_64-linux-gnu"),
                                                    interfaces.c.Strings.New_String("--device-as-default-execution-space"));

   type arguments is array(Positive range <>) of System.Address;

   type kernel is tagged record
      name : Ada.Strings.Unbounded.Unbounded_String;
      str : Ada.Strings.Unbounded.Unbounded_String;
      ptx : Interfaces.c.Strings.chars_ptr;
      ptx_size : aliased stddef_h.size_t;
      func : cuda_h.CUfunction;
   end record;
   function init(name, str : String) return kernel;

   procedure exec(self : kernel; num : Integer; args : arguments);
   procedure exec(self : kernel; blocks,threads : Integer; args : arguments);

   package hashed_maps is new ada.Containers.Indefinite_Hashed_Maps(Key_Type        => ada.Strings.Unbounded.Unbounded_String,
                                                                    Element_Type    => kernel,
                                                                    Hash => ada.Strings.Unbounded.Hash,
                                                                    Equivalent_Keys => "=",
                                                                    "="  => "=");
   map : hashed_maps.Map;
   procedure compile;


end cuda.driver;
