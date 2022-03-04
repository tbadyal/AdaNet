with ada.Numerics.Elementary_Functions; use ada.Numerics.Elementary_Functions;
with Ada.Numerics.Float_Random;

private package tensors.random is

   gen : Ada.Numerics.Float_Random.Generator;


   procedure uniform(self : in out tensor_t; gain : Float := Sqrt(2.0));
   procedure normal(self : in out tensor_t; gain : Float := Sqrt(2.0));


end tensors.random;
