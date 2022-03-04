with System;
with stddef_h;
with cudnn_h;
with storage;
with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;

package tensors is

   type randType_t is (UNIFORM,NORMAL);
   type randSource_t is (CPU,GPU);

   type tensor_t is new storage.storage_t with record
      d_address : System.Address;
      desc : cudnn_h.cudnnTensorDescriptor_t;
      fdesc : cudnn_h.cudnnFilterDescriptor_t;
   end record;
   function init(n,c,h,w : Positive := 1) return tensor_t;
   procedure free(self : in out tensor_t);
   function rand(n,c,h,w : Positive := 1; randType : randType_t := UNIFORM; randSource : randSource_t := GPU) return tensor_t;
   function fill(n,c,h,w : Positive := 1; value : float) return tensor_t;
   function ones(n,c,h,w : Positive := 1) return tensor_t;
   function zeros(n,c,h,w : Positive := 1) return tensor_t;
   procedure copy(self : tensor_t; oth : storage.storage_t);

   procedure to_device(self : tensor_t);
   procedure to_host(self : tensor_t);
   procedure memset(self : tensor_t; val : Integer);
   procedure memcopy(self, oth : tensor_t);

   function one_hot_decode(self : tensor_t; n : Positive) return Integer;
   procedure one_hot_encode(self : tensor_t; n : Positive; val : Integer);

   procedure add(self : tensor_t; oth : tensor_t);
   procedure sub(self : tensor_t; oth : tensor_t);
   procedure mul(self : tensor_t; oth : tensor_t);
   procedure div(self : tensor_t; oth : tensor_t);
   procedure pow(self : tensor_t; oth : Integer);
   procedure sqrt(self : tensor_t);

   procedure add(self : tensor_t; oth : Float);
   procedure sub(self : tensor_t; oth : Float);
   procedure mul(self : tensor_t; oth : Float);
   procedure div(self : tensor_t; oth : Float);


   procedure save(self : tensor_t; Stream : Stream_Access);

   procedure load(self : tensor_t; Stream : Stream_Access);

   procedure copy(self : tensor_t; oth : tensor_t);

   procedure add_ex(self : tensor_t; oth : tensor_t);
   procedure sub_ex(self : tensor_t; oth : tensor_t);
   procedure mul_ex(self : tensor_t; oth : tensor_t);
   procedure div_ex(self : tensor_t; oth : tensor_t);

   procedure add(self : tensor_t; r,g,b : float);
   procedure sub(self : tensor_t; r,g,b : float);
   procedure mul(self : tensor_t; r,g,b : float);
   procedure div(self : tensor_t; r,g,b : float);

   procedure read(self : in out tensor_t; files : storage.files_t);
   procedure show(self : tensor_t; msec : Integer := 0; winName : String := "AdaNet");
   procedure print(self : tensor_t);
   procedure resize(self, oth : in out tensor_t);

   procedure plot(x : tensor_t);
   procedure plot(x,y : tensor_t);


end tensors;
