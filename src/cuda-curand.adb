with System.Address_To_Access_Conversions;
with ada.Calendar.Conversions;
with interfaces.c.Extensions;

package body cuda.curand is

   gen : curand_h.curandGenerator_t;

   package Address_To_Access is new System.Address_To_Access_Conversions(Object => float);

   procedure uniform(self : in out tensors.tensor_t; gain : Float := Sqrt(2.0)) is
      num : Positive := (if self.num mod 2 /= 0 then self.num + 1 else self.num);
      receptive_field : Positive := self.h * self.w;
      fan_in : Positive := self.c * receptive_field;
      fan_out : Positive := self.n * receptive_field;
      mean : Float := 0.0;
      std : Float := gain * Sqrt(2.0 / float(fan_in + fan_out));
      bounds : Float := Sqrt(3.0) * std;
   begin

      checkCURAND(curand_h.curandGenerateUniform(generator => gen,
                                                 outputPtr => Address_To_Access.To_Pointer(self.d_address),
                                                 num       => stddef_h.size_t(num)));

      self.mul(2.0);
      self.mul(bounds);
      self.sub(bounds);


   end uniform;


   procedure normal(self : in out tensors.tensor_t; gain : Float := Sqrt(2.0)) is
      num : Positive := (if self.num mod 2 /= 0 then self.num + 1 else self.num);
      receptive_field : Positive := self.h * self.w;
      fan_in : Positive := self.c * receptive_field;
      fan_out : Positive := self.n * receptive_field;
      mean : Float := 0.0;
      std : Float := gain * Sqrt(2.0 / float(fan_in + fan_out));
   begin

      checkCURAND(curand_h.curandGenerateNormal(generator => gen,
                                                outputPtr => Address_To_Access.To_Pointer(self.d_address),
                                                n         => stddef_h.size_t(num),
                                                mean      => 0.0,
                                                stddev    => std));


   end normal;


begin

   checkCURAND(curand_h.curandCreateGenerator(generator => gen'Address,
                                              rng_type  => curand_h.CURAND_RNG_PSEUDO_DEFAULT));

   checkCURAND(curand_h.curandSetPseudoRandomGeneratorSeed(generator => gen,
                                                           seed      => interfaces.c.Extensions.unsigned_long_long(
                                                             ada.Calendar.Conversions.To_Unix_Time(
                                                               ada.Calendar.Clock))));



end cuda.curand;
