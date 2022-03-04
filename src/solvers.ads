with tensors; use tensors;
with cuda.driver;
with ada.Strings.Unbounded; use ada.Strings.Unbounded;
with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;

package solvers is

   type adam_t is tagged record
      alpha : Float := 1.0e-3;
      beta1 : Float := 0.9;
      beta2 : Float := 0.999;
      epsilon : Float := 1.0e-8;
      wd : Float := 1.0e-4;
      t : Positive := 1;
      f : tensor_t;
      df : tensor_t;
      Vdf : tensor_t;
      Sdf : tensor_t;
   end record;

   procedure Init(self : in out adam_t; f, df : tensor_t);

   procedure step(self : in out adam_t);

end solvers;
