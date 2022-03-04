with tensors;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;

package cuda.curand is

   procedure uniform(self : in out tensors.tensor_t; gain : Float := Sqrt(2.0));

   procedure normal(self : in out tensors.tensor_t; gain : Float := Sqrt(2.0));

end cuda.curand;
