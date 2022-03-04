package storage.image is

   type image_t(n,c,h,w,z : Positive) is tagged record
      image : storage_t := init(n,c,h,w);
      label : storage_t := init(n,z,1,1);
   end record;

   procedure save(self : image_t; Stream : Stream_Access);
   procedure load(self : image_t; Stream : Stream_Access);
   procedure free(self : in out image_t);

end storage.image;
