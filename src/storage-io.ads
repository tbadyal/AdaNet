private package storage.io is

   procedure read(self : in out storage_t; files : files_t);
   procedure show(self : storage_t; msec : Integer := 0; winName : String := "AdaNet");
   procedure resize(self, oth : in out storage_t);

end storage.io;
