package storage.stream is

   type stream_t is tagged limited record
      name : ada.Strings.Unbounded.Unbounded_String;
      file : ada.Streams.Stream_IO.File_Type;
      mode : ada.Streams.Stream_IO.File_Mode;
      hnd : ada.Streams.Stream_IO.Stream_Access;
   end record;

   function get_inStream(file : String) return stream_t;
   function get_outStream(file : String) return stream_t;
   procedure close_Stream(self : in out stream_t);
   function end_of_stream(self : stream_t) return Boolean;
end storage.stream;
