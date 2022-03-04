with ada.Unchecked_Deallocation;
with ada.Assertions; use ada.Assertions;
with ada.Text_IO; use Ada.Text_IO;
with ada.Integer_Text_IO; use ada.Integer_Text_IO;
with ada.Float_Text_IO; use ada.Float_Text_IO;
with storage.io;
with Ada.Numerics.Elementary_Functions;


package body storage is

   procedure free is new ada.Unchecked_Deallocation(Object => data_t,
                                                    Name   => data_acc);


   function idx(self : storage_t; n,c,h,w : Positive := 1) return reference_type is
   begin
      Assert(n<=self.n and c<=self.c and h<=self.h and w<=self.w,"Invalid Dimetions");

      return reference_type'(element => self.data(
                               (n-1)*self.c*self.h*self.w
                             + (c-1)*self.h*self.w
                             + (h-1)*self.w
                             +  w)'access);
   end idx;

   function init(n,c,h,w : Positive := 1) return storage_t is
   begin
      return self : storage_t do
         self.n := n;
         self.c := c;
         self.h := h;
         self.w := w;
         self.data := new data_t(1..n*c*h*w);
         self.data.all := (others => 0.0);
      end return;
   end init;

   procedure free(self : in out storage_t) is
   begin
      free(self.data);
   end free;

   function is_empty(self : storage_t) return boolean is
   begin
      if self.data = null then
         return True;
      else
         return False;
      end if;
   end is_empty;

   function num(self : storage_t) return integer is (self.n*self.c*self.h*self.w);

   function size(self : storage_t) return stddef_h.size_t is
      use type stddef_h.size_t;
   begin
      return self.data.all'size / 8;
   end size;

   function h_address(self : storage_t) return System.Address is
   begin
      return self.data.all'Address;
   end h_address;

   procedure fill(self : in out storage_t; value : float) is
   begin
      self.data.all := (others => value);
   end fill;

   procedure reshape(self : in out storage_t; n,c,h,w : Positive := 1) is
   begin
      Assert(n*c*h*w = self.n*self.c*self.h*self.w ,"Invalid reshape dimentions");
      self.n := n;
      self.c := c;
      self.h := h;
      self.w := w;
   end reshape;

   procedure save(self : storage_t; Stream : Stream_Access) is
   begin

      data_t'Write(stream, self.data.all);

   end save;

   procedure load(self : storage_t; Stream : Stream_Access) is
   begin

      data_t'Read(stream, self.data.all);

   end load;

   procedure dsc(self : storage_t) is
   begin
      Put_Line(" n:" & self.n'img & " c:" & self.c'img & " h:" & self.h'img & " w:" & self.w'img);
   end dsc;

   procedure print(self : storage_t) is
   begin
      self.dsc;
      for n in 1..self.n loop
         for c in 1..self.c loop
            for h in 1..self.h loop
               for w in 1..self.w loop
                  Put(self.idx(n,c,h,w), exp=>0);
                  Put(" ");
               end loop;
               Put(" ");
            end loop;
            Put(" ");
         end loop;
         New_Line;
      end loop;

   end print;

   procedure read(self : in out storage_t; files : files_t) is
   begin
      storage.io.read(self,files);
   end read;

   procedure show(self : storage_t; msec : Integer := 0; winName : String := "AdaNet") is
   begin
      storage.io.show(self,msec,winName);
   end show;

   procedure resize(self, oth : in out storage_t) is
   begin
      storage.io.resize(self,oth);
   end resize;

   procedure add(self : storage_t; oth : storage_t) is
   begin
      Assert(self.n = oth.n and self.c = oth.c and self.h = oth.h and self.w = oth.w,"Mismatching Dimentions");
      for i in self.data'Range loop
         self.data(i) := self.data(i) + oth.data(i);
      end loop;

   end add;

   procedure sub(self : storage_t; oth : storage_t) is
   begin
      Assert(self.n = oth.n and self.c = oth.c and self.h = oth.h and self.w = oth.w,"Mismatching Dimentions");
      for i in self.data'Range loop
         self.data(i) := self.data(i) - oth.data(i);
      end loop;

   end sub;

   procedure mul(self : storage_t; oth : storage_t)  is
   begin
      Assert(self.n = oth.n and self.c = oth.c and self.h = oth.h and self.w = oth.w,"Mismatching Dimentions");
      for i in self.data'Range loop
         self.data(i) := self.data(i) * oth.data(i);
      end loop;

   end mul;

   procedure div(self : storage_t; oth : storage_t)  is
   begin
      Assert(self.n = oth.n and self.c = oth.c and self.h = oth.h and self.w = oth.w,"Mismatching Dimentions");
      for i in self.data'Range loop
         self.data(i) := self.data(i) / oth.data(i);
      end loop;

   end div;

   procedure pow(self : storage_t; oth : Integer)  is
   begin
      for i in self.data'Range loop
         self.data(i) := self.data(i) ** oth;
      end loop;

   end pow;

   procedure sqrt(self : storage_t)  is
   begin
      for i in self.data'Range loop
         self.data(i) := Ada.Numerics.Elementary_Functions.Sqrt(self.data(i));
      end loop;

   end sqrt;

   procedure add(self : storage_t; oth : Float) is
   begin
      for i in self.data'Range loop
         self.data(i) := self.data(i) + oth;
      end loop;

   end add;

   procedure sub(self : storage_t; oth : Float) is
   begin
      for i in self.data'Range loop
         self.data(i) := self.data(i) - oth;
      end loop;

   end sub;

   procedure mul(self : storage_t; oth : Float)  is
   begin
      for i in self.data'Range loop
         self.data(i) := self.data(i) * oth;
      end loop;

   end mul;

   procedure div(self : storage_t; oth : Float)  is
   begin
      for i in self.data'Range loop
         self.data(i) := self.data(i) / oth;
      end loop;

   end div;

   procedure copy(self : storage_t; oth : storage_t) is
   begin
      Assert(self.n = oth.n and self.c = oth.c and self.h = oth.h and self.w = oth.w,"Mismatching Dimentions");
      for i in self.data'Range loop
         self.data(i) := oth.data(i);
      end loop;
   end copy;


end storage;
