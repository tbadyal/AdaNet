with ada.Sequential_IO;
with ada.Containers.Indefinite_Vectors;
with tensors; use tensors;
with storage.image; use storage.image;
with ada.Strings.Unbounded;
with Ada.Numerics.Discrete_Random;

package data.cifar10 is

   package images_vector_pkg is new ada.Containers.Indefinite_Vectors(Positive, image_t);
   package random_pkg is new Ada.Numerics.Discrete_Random(Result_Subtype => Integer);

   gen : random_pkg.Generator;

   type vector_t(n,c,h,w,z : Positive) is limited new dataset_t with record
      images : images_vector_pkg.Vector;
      mini_batches : images_vector_pkg.Vector;
   end record;

   procedure init(self : in out vector_t; file : String);
   procedure make_minibatches(self : in out vector_t);
   procedure shuffle(self : in out vector_t);
   procedure free(self : in out vector_t);
   function mean(self : vector_t) return dim3_t;
   function std(self : vector_t; mean_p : dim3_t) return dim3_t;


end data.cifar10;
