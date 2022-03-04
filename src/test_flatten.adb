with tensors;
with cuda.cudnn.make;

procedure test_flatten is
   x : tensors.tensor_t := tensors.rand(2,3,2,2);
   dy : tensors.tensor_t := tensors.rand(2,12);
   flat : cuda.cudnn.flatten_t := cuda.cudnn.make.flatten;
begin
   flat.init(x);
   flat.ibwd(dy);

   flat.x.print;
   flat.dy.print;

   flat.fwd;

   flat.y.print;

   flat.bwd;

   flat.dx.print;

end test_flatten;
