with Ada.Text_IO;
with ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with gnat.Traceback.Symbolic;
with cuda.cudnn.make;
with tensors;
with storage;

procedure test_pool is
   img : storage.files_t := (To_Unbounded_String("/home/tushar/gprprojects/adanet2/img/test1.png"),
                               To_Unbounded_String("/home/tushar/gprprojects/adanet2/img/test.png"));
   x : tensors.tensor_t := tensors.init(2,3,64,64);
   pool : cuda.cudnn.pooling_t :=
     cuda.cudnn.make.pooling(cuda.cudnn.MAX_POOL,2,2,2, padding => cuda.cudnn.VALID);
begin

   pool.init(x);

   x.read(img);

   pool.fwd;
   pool.x.dsc;
   pool.y.dsc;

   pool.y.show;
exception
   when e : others =>
      ada.Text_IO.Put_Line(gnat.Traceback.Symbolic.Symbolic_Traceback(e));
      raise;

end test_pool;
