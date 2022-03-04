with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with tensors;
with cuda.driver.kernels;
with System;
with cuda.io;
with Interfaces.C;
with ada.Text_IO;

use type Interfaces.c.unsigned_long;

package body solvers is


   procedure Init(self : in out adam_t; f, df : tensor_t) is
      begin
      self.f := f;
      self.df := df;
      self.Vdf := zeros(f.n,f.c,f.w,f.h);
      self.Sdf := zeros(f.n,f.c,f.w,f.h);
   end Init;

   procedure step(self : in out adam_t) is

      args : cuda.driver.arguments := (self.f.num'Address,
                                       self.f.d_address'Address,
                                       self.df.d_address'Address,
                                       self.vdf.d_address'Address,
                                       self.sdf.d_address'Address,
                                       self.t'Address,
                                       self.alpha'Address,
                                       self.beta1'Address,
                                       self.beta2'Address,
                                       self.epsilon'Address,
                                       self.wd'Address);

   begin

      cuda.driver.kernels.adam_kernel.exec(num  => self.f.num,
                                           args      => args);

      self.t := self.t + 1;

   end step;

end solvers;
