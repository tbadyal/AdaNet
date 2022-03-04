with storage; use storage;

package data is

   type dim3_t is record
      r,g,b : Float;
   end record;

   type dataset_t is limited interface;

   procedure init(self : in out dataset_t; file : String) is abstract;
   procedure make_minibatches(self : in out dataset_t) is abstract;
   procedure shuffle(self : in out dataset_t) is abstract;
   procedure free(self : in out dataset_t) is abstract;
   function mean(self : dataset_t) return dim3_t is abstract;
   function std(self : dataset_t; mean_p : dim3_t) return dim3_t is abstract;


end data;
