with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Assertions; use Ada.Assertions;
with Ada.Sequential_IO;
with storage.stream;
with Ada.Numerics.Elementary_Functions;
with Ada.Calendar.Conversions;

package body data.cifar10 is

   procedure init(self : in out vector_t; file : String) is

      stream : storage.stream.stream_t := storage.stream.get_inStream(file);


   begin

      while not stream.end_of_stream loop
         declare
            img : image_t(1,self.c,self.h,self.w,self.z);
         begin
            img.load(stream.hnd);

            self.images.Append(img);
         end;
      end loop;

      stream.close_Stream;

   end init;

   procedure make_minibatches(self : in out vector_t) is
      csr : images_vector_pkg.Cursor;
   begin
      self.free;
      self.mini_batches.Clear;
      csr := self.images.First;

      for i in 1..Integer(self.images.Length)/self.n loop

         declare
            mini_batch : image_t(self.n,self.c,self.h,self.w,self.z);
            start_image_idx : Positive := 1;
            inc_image_idx : Positive := self.c*self.h*self.w;
            end_image_idx : Positive := self.c*self.h*self.w;
            start_label_idx : Positive := 1;
            inc_label_idx : Positive := self.z;
            end_label_idx : Positive := self.z;
         begin

            for i in 1..self.n loop

               if images_vector_pkg.Has_Element(csr) then
                  mini_batch.image.data(start_image_idx..end_image_idx) := self.images(csr).image.data.all;
                  mini_batch.label.data(start_label_idx..end_label_idx) := self.images(csr).label.data.all;

                  start_image_idx := start_image_idx + inc_image_idx;
                  end_image_idx := end_image_idx + inc_image_idx;
                  start_label_idx := start_label_idx + inc_label_idx;
                  end_label_idx := end_label_idx + inc_label_idx;
               end if;

               images_vector_pkg.Next(csr);
            end loop;

            self.mini_batches.Append(mini_batch);

         end;

      end loop;

   end make_minibatches;

   procedure shuffle(self : in out vector_t) is
      j : Integer;
   begin
      for i in reverse self.images.First_Index ..self.images.Last_Index loop
         j := (random_pkg.Random(gen) mod i) + 1;
         self.images.Swap(i,j);
      end loop;
   end shuffle;

   procedure free(self : in out vector_t) is
   begin
      for mini_batch of self.mini_batches loop
         mini_batch.free;
      end loop;
   end free;


   function mean(self : vector_t) return dim3_t is
   begin
      return mean_l : dim3_t do
         for c in 1..3 loop

            declare
               total_batch : float := 0.0;
            begin
               for img of self.images loop

                  declare
                     total_image : float := 0.0;
                  begin
                     for n in 1..img.image.n loop
                        for h in 1..img.image.h loop
                           for w in 1..img.image.w loop
                              total_image := total_image + img.image.idx(n,c,h,w);
                           end loop;
                        end loop;
                     end loop;
                     total_batch := total_batch + total_image / float(img.image.n*img.image.h*img.image.w);
                  end;

               end loop;
               case c is
                  when 1 =>
                     mean_l.r := total_batch / float(self.images.length);
                  when 2 =>
                     mean_l.g := total_batch / float(self.images.length);
                  when 3 =>
                     mean_l.b := total_batch / float(self.images.length);
               end case;
            end;

         end loop;
      end return;
   end mean;

   function std(self : vector_t; mean_p : dim3_t) return dim3_t is
   begin
      return std_l : dim3_t do
         for c in 1..3 loop

            declare
               total_batch : float := 0.0;
            begin
               for img of self.images loop

                  declare
                     total_image : float := 0.0;
                  begin
                     for n in 1..img.image.n loop
                        for h in 1..img.image.h loop
                           for w in 1..img.image.w loop
                              case c is
                                 when 1 =>
                                    total_image := total_image + (img.image.idx(n,c,h,w) - mean_p.r)**2;
                                 when 2 =>
                                    total_image := total_image + (img.image.idx(n,c,h,w) - mean_p.g)**2;
                                 when 3 =>
                                    total_image := total_image + (img.image.idx(n,c,h,w) - mean_p.b)**2;
                              end case;
                           end loop;
                        end loop;
                     end loop;
                     total_batch := total_batch + total_image / float(img.image.n*img.image.h*img.image.w);
                  end;

               end loop;
               case c is
                  when 1 =>
                     std_l.r := Ada.Numerics.Elementary_Functions.Sqrt(total_batch / float(self.images.length));
                  when 2 =>
                     std_l.g := Ada.Numerics.Elementary_Functions.Sqrt(total_batch / float(self.images.length));
                  when 3 =>
                     std_l.b := Ada.Numerics.Elementary_Functions.Sqrt(total_batch / float(self.images.length));
               end case;
            end;

         end loop;
      end return;
   end std;

begin

   random_pkg.Reset(gen, Integer(Ada.Calendar.Conversions.To_Unix_Time(Ada.Calendar.Clock)));

end data.cifar10;
