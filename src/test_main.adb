with Ada.Text_IO;
with gnat.Traceback.Symbolic;
with data.cifar10;
with tensors;
with storage;
with cuda.cudnn.make;
with ada.Float_Text_IO;
with cuda.util;
with ada.Containers.Vectors;

procedure test_main is

   package float_vector_pkg is new ada.Containers.Vectors(Positive,Float);

   loss,accuracy : float_vector_pkg.Vector;

   function avg(self : float_vector_pkg.Vector) return float is
   begin
      return rval : float := 0.0 do
         for i of self loop
            rval := rval + i;
         end loop;
         rval := rval/float(self.Length);
      end return;
   end avg;

   training_set : data.cifar10.vector_t(32,3,64,64,10);
   validation_set : data.cifar10.vector_t(32,3,64,64,10);

   x : tensors.tensor_t := tensors.init(32,3,64,64);
   dy : tensors.tensor_t := tensors.init(32,10,1,1);
   m : data.dim3_t;
   s : data.dim3_t;

   net : cuda.cudnn.sequential_t;



begin
   training_set.init("../cifar10/prep/cifar10_train_images.bin");
   validation_set.init("../cifar10/prep/cifar10_test_images.bin");

   m := training_set.mean;
   s := training_set.std(m);

   net.add(cuda.cudnn.make.convolution(cuda.cudnn.CROSS_CORRELATION,64,7,7,2));
   net.add(cuda.cudnn.make.batchnorm(cuda.cudnn.SPATIAL));
   net.add(cuda.cudnn.make.activation(cuda.cudnn.RELU));
   net.add(cuda.cudnn.make.pooling(cuda.cudnn.MAX_POOL,3,3,2));

   net.add(cuda.cudnn.make.convolutional_block(64,64,256,stride => 1));
   net.add(cuda.cudnn.make.identity_block(64,64,256));
   net.add(cuda.cudnn.make.identity_block(64,64,256));

   net.add(cuda.cudnn.make.convolutional_block(128,128,512,stride => 2));
   net.add(cuda.cudnn.make.identity_block(128,128,512));
   net.add(cuda.cudnn.make.identity_block(128,128,512));
   net.add(cuda.cudnn.make.identity_block(128,128,512));

   net.add(cuda.cudnn.make.convolutional_block(256,256,1024,stride => 2));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));

   net.add(cuda.cudnn.make.convolutional_block(512,512,2048,stride => 2));
   net.add(cuda.cudnn.make.identity_block(512,512,2048));
   net.add(cuda.cudnn.make.identity_block(512,512,2048));


   net.add(cuda.cudnn.make.pooling(cuda.cudnn.AVG_INC_PAD,2,2, padding=>cuda.cudnn.SAME));

   net.add(cuda.cudnn.make.flatten);
   net.add(cuda.cudnn.make.fullyconnected(10));
   net.add(cuda.cudnn.make.softmax(cuda.cudnn.INSTANCE));

   net.add(cuda.cudnn.make.crossentropyloss);


   net.init(x,dy);

   for i in 1..100 loop

      training_set.shuffle;
      training_set.make_minibatches;

      declare
         idx : Positive := 1;
      begin

      for mini_batch of training_set.mini_batches loop

         while cuda.util.get_gpu_temp >= 85 loop
            delay 30.0;
         end loop;

      x.copy(mini_batch.image);
      dy.copy(mini_batch.label);

      x.sub(m.r,m.g,m.b);
      x.div(s.r,s.g,s.b);

      net.fwd;
      net.bwd;
         net.upd;

         loss.Append(net.loss);
         accuracy.Append(net.accuracy);

         if idx mod 10 = 0 then
            ada.Text_IO.put("#");
         end if;


         end loop;

      end;

      ada.Text_IO.put(i'img&" -> ");
      ada.Float_Text_IO.put(avg(loss), exp=>0);
      ada.Text_IO.put(" - ");
      ada.Float_Text_IO.put(avg(accuracy), exp=>0);


      loss.Clear;
      accuracy.Clear;

      validation_set.shuffle;
      validation_set.make_minibatches;

      for mini_batch of validation_set.mini_batches loop

         while cuda.util.get_gpu_temp >= 85 loop
            delay 30.0;
         end loop;

      x.copy(mini_batch.image);
      dy.copy(mini_batch.label);

      x.sub(m.r,m.g,m.b);
      x.div(s.r,s.g,s.b);

      net.fwd;

         loss.Append(net.loss);
         accuracy.Append(net.accuracy);


      end loop;

      ada.Text_IO.put(" - ");
      ada.Float_Text_IO.put(avg(loss), exp=>0);
      ada.Text_IO.put(" - ");
      ada.Float_Text_IO.put(avg(accuracy), exp=>0);

      loss.Clear;
      accuracy.Clear;

      ada.Text_IO.new_line;

   end loop;



exception
   when e : others =>
      ada.Text_IO.Put_Line(gnat.Traceback.Symbolic.Symbolic_Traceback(e));
      raise;

end test_main;
