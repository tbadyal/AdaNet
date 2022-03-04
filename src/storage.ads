with stddef_h;
with System;
with ada.Streams.Stream_IO; use ada.Streams.Stream_IO;
with ada.Strings.Unbounded; use ada.Strings.Unbounded;

package storage is

   type files_t is array(Positive range <>) of Unbounded_String;

   type data_t is array(Positive range <>) of aliased float;
   type data_acc is access data_t;
   type reference_type(element : not null access float) is null record
     with Implicit_Dereference => element;

   type storage_t is tagged record
      n,c,h,w : Positive := 1;
      data : data_acc;
   end record;


   function idx(self : storage_t; n,c,h,w : Positive := 1) return reference_type with inline;
   function init(n,c,h,w : Positive := 1) return storage_t;
   procedure fill(self : in out storage_t; value : float);
   procedure free(self : in out storage_t);
   function is_empty(self : storage_t) return boolean;
   function num(self : storage_t) return integer;
   function size(self : storage_t) return stddef_h.size_t;
   function h_address(self : storage_t) return System.Address;
   procedure reshape(self : in out storage_t; n,c,h,w : Positive := 1);
   procedure save(self : storage_t; Stream : Stream_Access);
   procedure load(self : storage_t; Stream : Stream_Access);
   procedure dsc(self : storage_t);
   procedure print(self : storage_t);
   procedure read(self : in out storage_t; files : files_t);
   procedure show(self : storage_t; msec : Integer := 0; winName : String := "AdaNet");
   procedure resize(self, oth : in out storage_t);

   procedure add(self : storage_t; oth : storage_t);
   procedure sub(self : storage_t; oth : storage_t);
   procedure mul(self : storage_t; oth : storage_t);
   procedure div(self : storage_t; oth : storage_t);
   procedure pow(self : storage_t; oth : Integer);
   procedure sqrt(self : storage_t);

   procedure add(self : storage_t; oth : Float);
   procedure sub(self : storage_t; oth : Float);
   procedure mul(self : storage_t; oth : Float);
   procedure div(self : storage_t; oth : Float);

   procedure copy(self : storage_t; oth : storage_t);

end storage;
