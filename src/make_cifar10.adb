with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Sequential_IO;
with storage.image; use storage.image;
with storage.stream;
with data;
with ada.Text_IO;
with gnat.Traceback.Symbolic;
with ada.Containers.Indefinite_Vectors;

procedure make_cifar10 is

   n : Positive := 1;
   z : Positive := 10;
   c : Positive := 3;
   h1 : Positive := 32;
   w1 : Positive := 32;
   h2 : Positive := 32;
   w2 : Positive := 32;


   train_list_g : storage.files_t := (To_Unbounded_String("../cifar10/data/data_batch_1.bin"),
                                      To_Unbounded_String("../cifar10/data/data_batch_2.bin"),
                                      To_Unbounded_String("../cifar10/data/data_batch_3.bin"),
                                      To_Unbounded_String("../cifar10/data/data_batch_4.bin"),
                                      To_Unbounded_String("../cifar10/data/data_batch_5.bin"));

   test_list_g : storage.files_t := (1 => To_Unbounded_String("../cifar10/data/test_batch.bin"));


   train_images_g : constant String := "../cifar10/prep/cifar10_train_images.bin";
   test_images_g : constant String := "../cifar10/prep/cifar10_test_images.bin";

   procedure get_images(files_list : storage.files_t; file_images : String) is


      type byte_t is mod 2**8;

      type byte_array_t is array(1..c*h1*w1) of byte_t;

      type cifar_t is record
         label : byte_t;
         data :  byte_array_t;
      end record;

      package cifar_io is new ada.Sequential_IO(cifar_t);

      cifar : cifar_t;


      stream : storage.stream.stream_t := storage.stream.get_outStream(file => file_images);

      cifar_hnd : cifar_io.File_Type;
      idx : Integer := 1;
      image : image_t(n,c,h1,w1,z);
      image2 : image_t(n,c,h2,w2,z);
   begin

      for file of files_list loop
         cifar_io.Open(cifar_hnd, cifar_io.In_File, To_String(file));

         while not cifar_io.End_Of_File(cifar_hnd) loop

            cifar_io.Read(cifar_hnd, cifar);


            for i in cifar.data'Range loop
               image.image.data(i) := float(cifar.data(i))/255.0;
            end loop;

            image.label.data.all := (others => 0.0);
            image.label.idx(1,integer(cifar.label)+1,1,1) := 1.0;
            image.image.resize(image2.image);
            image2.label := image.label;
            image2.save(stream.hnd);
            ada.Text_IO.Put_Line(idx'img);
            idx := idx + 1;
         end loop;

         cifar_io.Close(cifar_hnd);
      end loop;
      stream.close_Stream;
   end get_images;

begin
   get_images(train_list_g,train_images_g);
   get_images(test_list_g,test_images_g);
exception
      when e : others =>
      ada.Text_IO.Put_Line(gnat.Traceback.Symbolic.Symbolic_Traceback(e));
      raise;
end make_cifar10;
