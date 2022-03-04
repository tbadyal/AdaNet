with ada.Text_IO; use Ada.Text_IO;
with ada.Integer_Text_IO; use ada.Integer_Text_IO;
with ada.Float_Text_IO; use ada.Float_Text_IO;
with ada.Assertions; use ada.Assertions;
with cuda.io;
with cuda.cudnn;
with cuda.curand;
with Interfaces.c;
with cuda.driver.kernels;
with tensors.random;
with storage.utils;

use cuda;

package body tensors is

   -----------------------------------------------------------------------------

   function init(n,c,h,w : Positive := 1) return tensor_t is
   begin
      return self : tensor_t do
         storage.storage_t(self) := storage.init(n,c,h,w);
         self.d_address := cuda.io.malloc(self.size);
         cuda.io.memset(self.d_address, self.size, 0);
         checkCUDNN(cudnn_h.cudnnCreateTensorDescriptor(self.desc'Address));
         checkCUDNN(cudnn_h.cudnnSetTensor4dDescriptor(tensorDesc => self.desc,
                                                       format     => cudnn_h.CUDNN_TENSOR_NCHW,
                                                       dataType   => cudnn_h.CUDNN_DATA_FLOAT,
                                                       n          => Interfaces.c.int(self.n),
                                                       c          => Interfaces.c.int(self.c),
                                                       h          => Interfaces.c.int(self.h),
                                                       w          => Interfaces.c.int(self.w)));
         checkCUDNN(cudnn_h.cudnnCreateFilterDescriptor(self.fdesc'Address));
         checkCUDNN(cudnn_h.cudnnSetFilter4dDescriptor(filterDesc => self.fdesc,
                                                       dataType   => cudnn_h.CUDNN_DATA_FLOAT,
                                                       format     => cudnn_h.CUDNN_TENSOR_NCHW,
                                                       k          => Interfaces.c.int(self.n),
                                                       c          => Interfaces.c.int(self.c),
                                                       h          => Interfaces.c.int(self.h),
                                                       w          => Interfaces.c.int(self.w)));
      end return;
   end init;


   procedure free(self : in out tensor_t) is
   begin
      storage.storage_t(self).free;
      cuda.io.free(self.d_address);
      cuda.checkCUDNN(cudnn_h.cudnnDestroyTensorDescriptor(self.desc));
      cuda.checkCUDNN(cudnn_h.cudnnDestroyFilterDescriptor(self.fdesc));
   end free;

   -----------------------------------------------------------------------------

   function rand(n,c,h,w : Positive := 1; randType : randType_t := UNIFORM; randSource : randSource_t := GPU) return tensor_t is
   begin
      return self : tensor_t := init(n,c,h,w) do
         case randType is
         when NORMAL =>
            case randSource is
               when GPU =>
                  cuda.curand.normal(self);
               when CPU =>
                  tensors.random.normal(self);
            end case;
         when UNIFORM =>
            case randSource is
               when GPU =>
                  cuda.curand.uniform(self);
               when CPU =>
                  tensors.random.uniform(self);
            end case;
         end case;
      end return;
   end rand;

   function fill(n,c,h,w : Positive := 1; value : float) return tensor_t is
   begin
      return self : tensor_t := init(n,c,h,w) do
         storage.storage_t(self).fill(value);
         self.to_device;
      end return;
   end fill;

   function ones(n,c,h,w : Positive := 1) return tensor_t is
   begin
      return fill(n,c,h,w,1.0);
   end ones;

   function zeros(n,c,h,w : Positive := 1) return tensor_t is
   begin
      return fill(n,c,h,w,0.0);
   end zeros;

   procedure copy(self : tensor_t; oth : storage.storage_t) is
   begin
      storage.storage_t(self).data.all := oth.data.all;
      self.to_device;
   end copy;

   -----------------------------------------------------------------------------

   procedure to_device(self : tensor_t) is
   begin
      cuda.io.to_device(self.h_address, self.d_address, self.size);
   end to_device;

   procedure to_host(self : tensor_t) is
   begin
      cuda.io.to_host(self.d_address, self.h_address, self.size);
   end to_host;

   procedure memset(self : tensor_t; val : Integer) is
   begin
       cuda.io.memset(self.d_address, self.size, val);
   end memset;

   procedure memcopy(self, oth : tensor_t) is
   begin
      cuda.io.memcopy(self.d_address, oth.d_address, self.size);
   end memcopy;

   function one_hot_decode(self : tensor_t; n : Positive) return Integer is
   begin
      self.to_host;
      return val : Integer do
         for n in 1..self.n loop
            for c in 1..self.c loop
               if float'Rounding(self.idx(n,c,1,1)) = 1.0 then
                  val := c-1;
               end if;
            end loop;
         end loop;
      end return;
   end one_hot_decode;

   procedure one_hot_encode(self : tensor_t; n : Positive; val : Integer) is
   begin
      for c in 1..self.c loop
         self.idx(n,c,1,1) := 0.0;
      end loop;
      self.idx(n,val+1,1,1) := 1.0;
      self.to_device;
   end one_hot_encode;

   -----------------------------------------------------------------------------

   procedure add(self : tensor_t; oth : tensor_t)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin

      cuda.driver.kernels.tensor_add.exec(self.num,args);

   end add;

   procedure sub(self : tensor_t; oth : tensor_t)
   is
      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin


      cuda.driver.kernels.tensor_sub.exec(self.num,args);

   end sub;

   procedure mul(self : tensor_t; oth : tensor_t)
   is
      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin


      cuda.driver.kernels.tensor_mul.exec(self.num,args);

   end mul;

   procedure div(self : tensor_t; oth : tensor_t)
   is
      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin


      cuda.driver.kernels.tensor_div.exec(self.num,args);

   end div;

   procedure pow(self : tensor_t; oth : Integer)
   is
      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth'Address);
   begin

      cuda.driver.kernels.tensor_pow.exec(self.num,args);
   end pow;

   procedure sqrt(self : tensor_t)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address);
   begin

      cuda.driver.kernels.tensor_sqrt.exec(self.num,args);
   end sqrt;

   procedure add(self : tensor_t; oth : Float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth'Address);
   begin

      cuda.driver.kernels.tensor_scal_add.exec(self.num,args);
   end add;

   procedure sub(self : tensor_t; oth : Float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth'Address);
   begin

      cuda.driver.kernels.tensor_scal_sub.exec(self.num,args);
   end sub;

  procedure mul(self : tensor_t; oth : Float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth'Address);
   begin

      cuda.driver.kernels.tensor_scal_mul.exec(self.num,args);
   end mul;

   procedure div(self : tensor_t; oth : Float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth'Address);
   begin

      cuda.driver.kernels.tensor_scal_div.exec(self.num,args);
   end div;


   procedure save(self : tensor_t; Stream : Ada.Streams.Stream_IO.Stream_Access) is
   begin

      self.to_host;
      storage.storage_t(self).save(stream);

   end save;

   procedure load(self : tensor_t; Stream : Ada.Streams.Stream_IO.Stream_Access) is
   begin

      storage.storage_t(self).load(stream);
      self.to_device;
   end load;

   procedure copy(self : tensor_t; oth : tensor_t)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin

      cuda.driver.kernels.tensor_copy.exec(self.num,args);

   end copy;

   procedure add_ex(self : tensor_t; oth : tensor_t)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin

      cuda.driver.kernels.tensor_add_ex.exec(self.num,args);

   end add_ex;

   procedure sub_ex(self : tensor_t; oth : tensor_t)
   is
      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin


      cuda.driver.kernels.tensor_sub_ex.exec(self.num,args);

   end sub_ex;

   procedure mul_ex(self : tensor_t; oth : tensor_t)
   is
      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin


      cuda.driver.kernels.tensor_mul_ex.exec(self.num,args);

   end mul_ex;

   procedure div_ex(self : tensor_t; oth : tensor_t)
   is
      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       oth.d_address'Address);
   begin


      cuda.driver.kernels.tensor_div_ex.exec(self.num,args);

   end div_ex;

   procedure add(self : tensor_t; r,g,b : float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       r'Address,
                                       g'Address,
                                       b'Address,
                                       self.n'Address,
                                       self.c'Address,
                                       self.h'Address,
                                       self.w'Address);
   begin

      cuda.driver.kernels.tensor_add_scal_ex.exec(self.num,args);

   end add;

   procedure sub(self : tensor_t; r,g,b : float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       r'Address,
                                       g'Address,
                                       b'Address,
                                       self.n'Address,
                                       self.c'Address,
                                       self.h'Address,
                                       self.w'Address);
   begin

      cuda.driver.kernels.tensor_sub_scal_ex.exec(self.num,args);

   end sub;

   procedure mul(self : tensor_t; r,g,b : float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       r'Address,
                                       g'Address,
                                       b'Address,
                                       self.n'Address,
                                       self.c'Address,
                                       self.h'Address,
                                       self.w'Address);
   begin

      cuda.driver.kernels.tensor_mul_scal_ex.exec(self.num,args);

   end mul;

   procedure div(self : tensor_t; r,g,b : float)
   is

      args : cuda.driver.arguments := (self.num'Address,
                                       self.d_address'Address,
                                       r'Address,
                                       g'Address,
                                       b'Address,
                                       self.n'Address,
                                       self.c'Address,
                                       self.h'Address,
                                       self.w'Address);
   begin

      cuda.driver.kernels.tensor_div_scal_ex.exec(self.num,args);

   end div;

   procedure read(self : in out tensor_t; files : storage.files_t) is
   begin
      storage.storage_t(self).read(files);
      self.to_device;
   end read;


   procedure show(self : tensor_t; msec : Integer := 0; winName : String := "AdaNet") is
   begin
      self.to_host;
      storage.storage_t(self).show(msec,winName);
   end show;

   procedure print(self : tensor_t) is
   begin
      self.to_host;
      storage.storage_t(self).print;
   end print;


   procedure resize(self, oth : in out tensor_t) is
   begin
      self.to_host;

      storage.storage_t(self).resize(storage.storage_t(oth));

      oth.to_device;

   end resize;

   procedure plot(x : tensor_t) is
   begin
      x.to_host;
      storage.utils.plot(storage.storage_t(x));
   end plot;

   procedure plot(x,y : tensor_t) is
   begin
      x.to_host;
      y.to_host;
      storage.utils.plot(storage.storage_t(x),storage.storage_t(y));
   end plot;

end tensors;
