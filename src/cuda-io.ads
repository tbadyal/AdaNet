with system;
with stddef_h;

package cuda.io is

   function malloc(size : stddef_h.size_t) return System.Address;

   procedure free(dx : System.Address);

   procedure to_device(x : System.Address; dx : System.Address; size : stddef_h.size_t);

   procedure to_host(dx : System.Address; x : System.Address; size : stddef_h.size_t);

   procedure memset(dx : System.Address; size : stddef_h.size_t; val : Integer);

   procedure memcopy(dx, dy : System.Address; size : stddef_h.size_t);

   function mallocManaged(size : stddef_h.size_t) return System.Address;

end cuda.io;
