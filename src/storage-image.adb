package body storage.image is

   procedure save(self : image_t; Stream : Stream_Access) is
   begin
      self.image.save(stream);
      self.label.save(stream);
   end save;

   procedure load(self : image_t; Stream : Stream_Access) is
   begin
      self.image.load(stream);
      self.label.load(stream);
   end load;

   procedure free(self : in out image_t) is
   begin
      self.image.free;
      self.label.free;
   end free;

end storage.image;
