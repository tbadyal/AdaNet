with data.cifar10;
with tensors;
with storage;
with Ada.Text_IO;
with ada.Float_Text_IO;
with gnat.Traceback.Symbolic;

procedure cifar10_test is
   training_set : data.cifar10.vector_t(10,3,32,32,10);
   test_set : data.cifar10.vector_t(10,3,32,32,10);
   x : tensors.tensor_t := tensors.init(10,3,32,32);
   l : tensors.tensor_t := tensors.init(10,10,1,1);
   m : data.dim3_t;
   s : data.dim3_t;
begin

   training_set.init("../cifar10/prep/cifar10_train_images.bin");
   test_set.init("../cifar10/prep/cifar10_test_images.bin");


   m := training_set.mean;
   s := training_set.std(m);


   training_set.shuffle;
   training_set.make_minibatches;


   for mini_batch of training_set.mini_batches loop



      x.copy(mini_batch.image);
      l.copy(mini_batch.label);

      x.sub(0.5,0.5,0.5);
      x.div(0.5,0.5,0.5);


      x.show(1);
      l.print;


   end loop;

   training_set.shuffle;
   training_set.make_minibatches;

   for mini_batch of training_set.mini_batches loop

      x.copy(mini_batch.image);
      l.copy(mini_batch.label);

      x.add(0.9,0.9,0.9);


      x.show(10);
      l.print;


   end loop;


exception
      when e : others =>
      ada.Text_IO.Put_Line(gnat.Traceback.Symbolic.Symbolic_Traceback(e));
      raise;
end cifar10_test;
