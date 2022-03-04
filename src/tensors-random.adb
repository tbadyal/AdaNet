with ada.Numerics.Elementary_Functions; use ada.Numerics; use ada.Numerics.Elementary_Functions;
with Ada.Calendar.Conversions; use Ada.Calendar.Conversions; use ada.Calendar;

package body tensors.random is


   procedure uniform(self : in out tensor_t; gain : Float := Sqrt(2.0)) is
      receptive_field : Positive := self.h * self.w;
      fan_in : Positive := self.c * receptive_field;
      fan_out : Positive := self.n * receptive_field;
      mean : Float := 0.0;
      std : Float := gain * Sqrt(2.0 / float(fan_in + fan_out));
      bounds : Float := Sqrt(3.0) * std;
   begin
      for i of self.data.all loop
         i := 2.0 * bounds * Ada.Numerics.Float_Random.Random(gen) - bounds;
      end loop;

      self.to_device;

   end uniform;

   procedure normal(self : in out tensor_t; gain : Float := Sqrt(2.0)) is
      receptive_field : Positive := self.h * self.w;
      fan_in : Positive := self.c * receptive_field;
      fan_out : Positive := self.n * receptive_field;
      mean : Float := 0.0;
      std : Float := gain * Sqrt(2.0 / float(fan_in + fan_out));
      bounds : Float := Sqrt(3.0) * std;
   begin

      for i of self.data.all loop
         declare
            r : Float := Ada.Numerics.Float_Random.Random(gen);
         begin
            i := mean + (std * Sqrt (-2.0 * Log (r, 10.0)) * Cos (2.0 * Pi * r));
               end;
      end loop;

      self.to_device;

   end normal;



begin

   Ada.Numerics.Float_Random.Reset(gen, Integer(To_Unix_Time(Clock)));

end tensors.random;
