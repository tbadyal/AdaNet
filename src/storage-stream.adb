package body storage.stream is

   function get_inStream(file : String) return stream_t is
   begin
      return self : stream_t do
         self.name := To_Unbounded_String(file);
         self.mode := ada.Streams.Stream_IO.In_File;
         ada.Streams.Stream_IO.Open(File => self.file,
                                    Mode => self.mode,
                                    Name => To_String(self.name),
                                    Form => "shared=yes");

         ada.Streams.Stream_IO.Reset(self.file);
         self.hnd := ada.Streams.Stream_IO.Stream(File => self.file);

      end return;
   end get_inStream;

   function get_outStream(file : String) return stream_t is
   begin
      return self : stream_t do
         self.name := To_Unbounded_String(file);
         self.mode := ada.Streams.Stream_IO.Out_File;
         begin
            ada.Streams.Stream_IO.Open(File => self.file,
                                       Mode => self.mode,
                                       Name => To_String(self.name),
                                       Form => "shared=yes");
         exception
            when ada.Streams.Stream_IO.Name_Error =>

               ada.Streams.Stream_IO.Create(File => self.file,
                                            Mode => self.mode,
                                            Name => To_String(self.name),
                                            Form => "shared=yes");
         end;
         ada.Streams.Stream_IO.Reset(self.file);
         self.hnd := ada.Streams.Stream_IO.Stream(File => self.file);

      end return;
   end get_outStream;

   procedure close_Stream(self : in out stream_t) is
   begin

      ada.Streams.Stream_IO.Close(File => self.file);
   end close_Stream;

   function end_of_stream(self : stream_t) return Boolean is
   begin
      return ada.Streams.Stream_IO.End_Of_File(File => self.file);
   end end_of_stream;

end storage.stream;
