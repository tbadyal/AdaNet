pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with stddef_h;
with Interfaces.C.Strings;
with xlocale_h;

package string_h is

   --  arg-macro: function strdupa (__extension__ ({ const char *__old := (s); size_t __len := strlen (__old) + 1; char *__new := (char *) __builtin_alloca (__len); (char *) memcpy (__new, __old, __len); })
   --    return __extension__ ({ const char *__old := (s); size_t __len := strlen (__old) + 1; char *__new := (char *) __builtin_alloca (__len); (char *) memcpy (__new, __old, __len); });
   --  arg-macro: function strndupa (__extension__ ({ const char *__old := (s); size_t __len := strnlen (__old, (n)); char *__new := (char *) __builtin_alloca (__len + 1); __new(__len) := Character'Val (0); (char *) memcpy (__new, __old, __len); })
   --    return __extension__ ({ const char *__old := (s); size_t __len := strnlen (__old, (n)); char *__new := (char *) __builtin_alloca (__len + 1); __new(__len) := Character'Val (0); (char *) memcpy (__new, __old, __len); });
  -- Copyright (C) 1991-2016 Free Software Foundation, Inc.
  --   This file is part of the GNU C Library.
  --   The GNU C Library is free software; you can redistribute it and/or
  --   modify it under the terms of the GNU Lesser General Public
  --   License as published by the Free Software Foundation; either
  --   version 2.1 of the License, or (at your option) any later version.
  --   The GNU C Library is distributed in the hope that it will be useful,
  --   but WITHOUT ANY WARRANTY; without even the implied warranty of
  --   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  --   Lesser General Public License for more details.
  --   You should have received a copy of the GNU Lesser General Public
  --   License along with the GNU C Library; if not, see
  --   <http://www.gnu.org/licenses/>.   

  -- *	ISO C99 Standard: 7.21 String handling	<string.h>
  --  

  -- Get size_t and NULL from <stddef.h>.   
  -- Tell the caller that we provide correct C++ prototypes.   
  -- Copy N bytes of SRC to DEST.   
   function memcpy
     (uu_dest : System.Address;
      uu_src : System.Address;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:42
   pragma Import (C, memcpy, "memcpy");

  -- Copy N bytes of SRC to DEST, guaranteeing
  --   correct behavior for overlapping strings.   

   function memmove
     (uu_dest : System.Address;
      uu_src : System.Address;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:46
   pragma Import (C, memmove, "memmove");

  -- Copy no more than N bytes of SRC to DEST, stopping when C is found.
  --   Return the position in DEST one byte past where C was copied,
  --   or NULL if C was not found in the first N bytes of SRC.   

   function memccpy
     (uu_dest : System.Address;
      uu_src : System.Address;
      uu_c : int;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:54
   pragma Import (C, memccpy, "memccpy");

  -- Set N bytes of S to C.   
   function memset
     (uu_s : System.Address;
      uu_c : int;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:62
   pragma Import (C, memset, "memset");

  -- Compare N bytes of S1 and S2.   
   function memcmp
     (uu_s1 : System.Address;
      uu_s2 : System.Address;
      uu_n : stddef_h.size_t) return int;  -- /usr/include/string.h:65
   pragma Import (C, memcmp, "memcmp");

  -- Search N bytes of S for C.   
   function memchr
     (uu_s : System.Address;
      uu_c : int;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:72
   pragma Import (C, memchr, "memchr");

   function memchr
     (uu_s : System.Address;
      uu_c : int;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:74
   pragma Import (C, memchr, "memchr");

  -- Search in S for C.  This is similar to `memchr' but there is no
  --   length limit.   

   function rawmemchr (uu_s : System.Address; uu_c : int) return System.Address;  -- /usr/include/string.h:101
   pragma Import (C, rawmemchr, "rawmemchr");

   function rawmemchr (uu_s : System.Address; uu_c : int) return System.Address;  -- /usr/include/string.h:103
   pragma Import (C, rawmemchr, "rawmemchr");

  -- Search N bytes of S for the final occurrence of C.   
   function memrchr
     (uu_s : System.Address;
      uu_c : int;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:112
   pragma Import (C, memrchr, "memrchr");

   function memrchr
     (uu_s : System.Address;
      uu_c : int;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:114
   pragma Import (C, memrchr, "memrchr");

  -- Copy SRC to DEST.   
   function strcpy (uu_dest : Interfaces.C.Strings.chars_ptr; uu_src : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:125
   pragma Import (C, strcpy, "strcpy");

  -- Copy no more than N characters of SRC to DEST.   
   function strncpy
     (uu_dest : Interfaces.C.Strings.chars_ptr;
      uu_src : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:128
   pragma Import (C, strncpy, "strncpy");

  -- Append SRC onto DEST.   
   function strcat (uu_dest : Interfaces.C.Strings.chars_ptr; uu_src : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:133
   pragma Import (C, strcat, "strcat");

  -- Append no more than N characters from SRC onto DEST.   
   function strncat
     (uu_dest : Interfaces.C.Strings.chars_ptr;
      uu_src : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:136
   pragma Import (C, strncat, "strncat");

  -- Compare S1 and S2.   
   function strcmp (uu_s1 : Interfaces.C.Strings.chars_ptr; uu_s2 : Interfaces.C.Strings.chars_ptr) return int;  -- /usr/include/string.h:140
   pragma Import (C, strcmp, "strcmp");

  -- Compare N characters of S1 and S2.   
   function strncmp
     (uu_s1 : Interfaces.C.Strings.chars_ptr;
      uu_s2 : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t) return int;  -- /usr/include/string.h:143
   pragma Import (C, strncmp, "strncmp");

  -- Compare the collated forms of S1 and S2.   
   function strcoll (uu_s1 : Interfaces.C.Strings.chars_ptr; uu_s2 : Interfaces.C.Strings.chars_ptr) return int;  -- /usr/include/string.h:147
   pragma Import (C, strcoll, "strcoll");

  -- Put a transformation of SRC into no more than N bytes of DEST.   
   function strxfrm
     (uu_dest : Interfaces.C.Strings.chars_ptr;
      uu_src : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t) return stddef_h.size_t;  -- /usr/include/string.h:150
   pragma Import (C, strxfrm, "strxfrm");

  -- The following functions are equivalent to the both above but they
  --   take the locale they use for the collation as an extra argument.
  --   This is not standardsized but something like will come.   

  -- Compare the collated forms of S1 and S2 using rules from L.   
   function strcoll_l
     (uu_s1 : Interfaces.C.Strings.chars_ptr;
      uu_s2 : Interfaces.C.Strings.chars_ptr;
      uu_l : xlocale_h.uu_locale_t) return int;  -- /usr/include/string.h:162
   pragma Import (C, strcoll_l, "strcoll_l");

  -- Put a transformation of SRC into no more than N bytes of DEST.   
   function strxfrm_l
     (uu_dest : Interfaces.C.Strings.chars_ptr;
      uu_src : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t;
      uu_l : xlocale_h.uu_locale_t) return stddef_h.size_t;  -- /usr/include/string.h:165
   pragma Import (C, strxfrm_l, "strxfrm_l");

  -- Duplicate S, returning an identical malloc'd string.   
   function strdup (uu_s : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:171
   pragma Import (C, strdup, "strdup");

  -- Return a malloc'd copy of at most N bytes of STRING.  The
  --   resultant string is terminated even if no null terminator
  --   appears before STRING[N].   

   function strndup (uu_string : Interfaces.C.Strings.chars_ptr; uu_n : stddef_h.size_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:179
   pragma Import (C, strndup, "strndup");

  -- Duplicate S, returning an identical alloca'd string.   
  -- Return an alloca'd copy of at most N bytes of string.   
  -- Find the first occurrence of C in S.   
   function strchr (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:211
   pragma Import (C, strchr, "strchr");

   function strchr (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:213
   pragma Import (C, strchr, "strchr");

  -- Find the last occurrence of C in S.   
   function strrchr (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:238
   pragma Import (C, strrchr, "strrchr");

   function strrchr (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:240
   pragma Import (C, strrchr, "strrchr");

  -- This function is similar to `strchr'.  But it returns a pointer to
  --   the closing NUL byte in case C is not found in S.   

   function strchrnul (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:267
   pragma Import (C, strchrnul, "strchrnul");

   function strchrnul (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:269
   pragma Import (C, strchrnul, "strchrnul");

  -- Return the length of the initial segment of S which
  --   consists entirely of characters not in REJECT.   

   function strcspn (uu_s : Interfaces.C.Strings.chars_ptr; uu_reject : Interfaces.C.Strings.chars_ptr) return stddef_h.size_t;  -- /usr/include/string.h:280
   pragma Import (C, strcspn, "strcspn");

  -- Return the length of the initial segment of S which
  --   consists entirely of characters in ACCEPT.   

   function strspn (uu_s : Interfaces.C.Strings.chars_ptr; uu_accept : Interfaces.C.Strings.chars_ptr) return stddef_h.size_t;  -- /usr/include/string.h:284
   pragma Import (C, strspn, "strspn");

  -- Find the first occurrence in S of any character in ACCEPT.   
   function strpbrk (uu_s : Interfaces.C.Strings.chars_ptr; uu_accept : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:290
   pragma Import (C, strpbrk, "strpbrk");

   function strpbrk (uu_s : Interfaces.C.Strings.chars_ptr; uu_accept : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:292
   pragma Import (C, strpbrk, "strpbrk");

  -- Find the first occurrence of NEEDLE in HAYSTACK.   
   function strstr (uu_haystack : Interfaces.C.Strings.chars_ptr; uu_needle : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:317
   pragma Import (C, strstr, "strstr");

   function strstr (uu_haystack : Interfaces.C.Strings.chars_ptr; uu_needle : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:319
   pragma Import (C, strstr, "strstr");

  -- Divide S into tokens separated by characters in DELIM.   
   function strtok (uu_s : Interfaces.C.Strings.chars_ptr; uu_delim : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:343
   pragma Import (C, strtok, "strtok");

  -- Divide S into tokens separated by characters in DELIM.  Information
  --   passed between calls are stored in SAVE_PTR.   

   --  skipped func __strtok_r

   function strtok_r
     (uu_s : Interfaces.C.Strings.chars_ptr;
      uu_delim : Interfaces.C.Strings.chars_ptr;
      uu_save_ptr : System.Address) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:354
   pragma Import (C, strtok_r, "strtok_r");

  -- Similar to `strstr' but this function ignores the case of both strings.   
   function strcasestr (uu_haystack : Interfaces.C.Strings.chars_ptr; uu_needle : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:362
   pragma Import (C, strcasestr, "strcasestr");

   function strcasestr (uu_haystack : Interfaces.C.Strings.chars_ptr; uu_needle : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:364
   pragma Import (C, strcasestr, "strcasestr");

  -- Find the first occurrence of NEEDLE in HAYSTACK.
  --   NEEDLE is NEEDLELEN bytes long;
  --   HAYSTACK is HAYSTACKLEN bytes long.   

   function memmem
     (uu_haystack : System.Address;
      uu_haystacklen : stddef_h.size_t;
      uu_needle : System.Address;
      uu_needlelen : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:377
   pragma Import (C, memmem, "memmem");

  -- Copy N bytes of SRC to DEST, return pointer to bytes after the
  --   last written byte.   

   --  skipped func __mempcpy

   function mempcpy
     (uu_dest : System.Address;
      uu_src : System.Address;
      uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:386
   pragma Import (C, mempcpy, "mempcpy");

  -- Return the length of S.   
   function strlen (uu_s : Interfaces.C.Strings.chars_ptr) return stddef_h.size_t;  -- /usr/include/string.h:394
   pragma Import (C, strlen, "strlen");

  -- Find the length of STRING, but scan at most MAXLEN characters.
  --   If no '\0' terminator is found in that many characters, return MAXLEN.   

   function strnlen (uu_string : Interfaces.C.Strings.chars_ptr; uu_maxlen : stddef_h.size_t) return stddef_h.size_t;  -- /usr/include/string.h:401
   pragma Import (C, strnlen, "strnlen");

  -- Return a string describing the meaning of the `errno' code in ERRNUM.   
   function strerror (uu_errnum : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:408
   pragma Import (C, strerror, "strerror");

  -- Reentrant version of `strerror'.
  --   There are 2 flavors of `strerror_r', GNU which returns the string
  --   and may or may not use the supplied temporary buffer and POSIX one
  --   which fills the string into the buffer.
  --   To use the POSIX version, -D_XOPEN_SOURCE=600 or -D_POSIX_C_SOURCE=200112L
  --   without -D_GNU_SOURCE is needed, otherwise the GNU version is
  --   preferred.   

  -- Fill BUF with a string describing the meaning of the `errno' code in
  --   ERRNUM.   

  -- If a temporary buffer is required, at most BUFLEN bytes of BUF will be
  --   used.   

   function strerror_r
     (uu_errnum : int;
      uu_buf : Interfaces.C.Strings.chars_ptr;
      uu_buflen : stddef_h.size_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:433
   pragma Import (C, strerror_r, "strerror_r");

  -- Translate error number to string according to the locale L.   
   function strerror_l (uu_errnum : int; uu_l : xlocale_h.uu_locale_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:440
   pragma Import (C, strerror_l, "strerror_l");

  -- We define this function always since `bzero' is sometimes needed when
  --   the namespace rules does not allow this.   

   --  skipped func __bzero

  -- Copy N bytes of SRC to DEST (like memmove, but args reversed).   
   procedure bcopy
     (uu_src : System.Address;
      uu_dest : System.Address;
      uu_n : stddef_h.size_t);  -- /usr/include/string.h:450
   pragma Import (C, bcopy, "bcopy");

  -- Set N bytes of S to 0.   
   procedure bzero (uu_s : System.Address; uu_n : stddef_h.size_t);  -- /usr/include/string.h:454
   pragma Import (C, bzero, "bzero");

  -- Compare N bytes of S1 and S2 (same as memcmp).   
   function bcmp
     (uu_s1 : System.Address;
      uu_s2 : System.Address;
      uu_n : stddef_h.size_t) return int;  -- /usr/include/string.h:457
   pragma Import (C, bcmp, "bcmp");

  -- Find the first occurrence of C in S (same as strchr).   
   function index (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:464
   pragma Import (C, index, "index");

   function index (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:466
   pragma Import (C, index, "index");

  -- Find the last occurrence of C in S (same as strrchr).   
   function rindex (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:492
   pragma Import (C, rindex, "rindex");

   function rindex (uu_s : Interfaces.C.Strings.chars_ptr; uu_c : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:494
   pragma Import (C, rindex, "rindex");

  -- Return the position of the first bit set in I, or 0 if none are set.
  --   The least-significant bit is position 1, the most-significant 32.   

   function ffs (uu_i : int) return int;  -- /usr/include/string.h:518
   pragma Import (C, ffs, "ffs");

  -- The following two functions are non-standard but necessary for non-32 bit
  --   platforms.   

   function ffsl (uu_l : long) return int;  -- /usr/include/string.h:523
   pragma Import (C, ffsl, "ffsl");

   function ffsll (uu_ll : Long_Long_Integer) return int;  -- /usr/include/string.h:524
   pragma Import (C, ffsll, "ffsll");

  -- Compare S1 and S2, ignoring case.   
   function strcasecmp (uu_s1 : Interfaces.C.Strings.chars_ptr; uu_s2 : Interfaces.C.Strings.chars_ptr) return int;  -- /usr/include/string.h:529
   pragma Import (C, strcasecmp, "strcasecmp");

  -- Compare no more than N chars of S1 and S2, ignoring case.   
   function strncasecmp
     (uu_s1 : Interfaces.C.Strings.chars_ptr;
      uu_s2 : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t) return int;  -- /usr/include/string.h:533
   pragma Import (C, strncasecmp, "strncasecmp");

  -- Again versions of a few functions which use the given locale instead
  --   of the global one.   

   function strcasecmp_l
     (uu_s1 : Interfaces.C.Strings.chars_ptr;
      uu_s2 : Interfaces.C.Strings.chars_ptr;
      uu_loc : xlocale_h.uu_locale_t) return int;  -- /usr/include/string.h:540
   pragma Import (C, strcasecmp_l, "strcasecmp_l");

   function strncasecmp_l
     (uu_s1 : Interfaces.C.Strings.chars_ptr;
      uu_s2 : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t;
      uu_loc : xlocale_h.uu_locale_t) return int;  -- /usr/include/string.h:544
   pragma Import (C, strncasecmp_l, "strncasecmp_l");

  -- Return the next DELIM-delimited token from *STRINGP,
  --   terminating it with a '\0', and update *STRINGP to point past it.   

   function strsep (uu_stringp : System.Address; uu_delim : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:552
   pragma Import (C, strsep, "strsep");

  -- Return a string describing the meaning of the signal number in SIG.   
   function strsignal (uu_sig : int) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:559
   pragma Import (C, strsignal, "strsignal");

  -- Copy SRC to DEST, returning the address of the terminating '\0' in DEST.   
   --  skipped func __stpcpy

   function stpcpy (uu_dest : Interfaces.C.Strings.chars_ptr; uu_src : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:564
   pragma Import (C, stpcpy, "stpcpy");

  -- Copy no more than N characters of SRC to DEST, returning the address of
  --   the last character written into DEST.   

   --  skipped func __stpncpy

   function stpncpy
     (uu_dest : Interfaces.C.Strings.chars_ptr;
      uu_src : Interfaces.C.Strings.chars_ptr;
      uu_n : stddef_h.size_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:572
   pragma Import (C, stpncpy, "stpncpy");

  -- Compare S1 and S2 as strings holding name & indices/version numbers.   
   function strverscmp (uu_s1 : Interfaces.C.Strings.chars_ptr; uu_s2 : Interfaces.C.Strings.chars_ptr) return int;  -- /usr/include/string.h:579
   pragma Import (C, strverscmp, "strverscmp");

  -- Sautee STRING briskly.   
   function strfry (uu_string : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:583
   pragma Import (C, strfry, "strfry");

  -- Frobnicate N bytes of S.   
   function memfrob (uu_s : System.Address; uu_n : stddef_h.size_t) return System.Address;  -- /usr/include/string.h:586
   pragma Import (C, memfrob, "memfrob");

  -- Return the file name within directory of FILENAME.  We don't
  --   declare the function if the `basename' macro is available (defined
  --   in <libgen.h>) which makes the XPG version of this function
  --   available.   

   function basename (uu_filename : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:594
   pragma Import (C, basename, "basename");

   function basename (uu_filename : Interfaces.C.Strings.chars_ptr) return Interfaces.C.Strings.chars_ptr;  -- /usr/include/string.h:596
   pragma Import (C, basename, "basename");

  -- When using GNU CC we provide some optimized versions of selected
  --   functions from this header.  There are two kinds of optimizations:
  --   - machine-dependent optimizations, most probably using inline
  --     assembler code; these might be quite expensive since the code
  --     size can increase significantly.
  --     These optimizations are not used unless the symbol
  --	__USE_STRING_INLINES
  --     is defined before including this header.
  --   - machine-independent optimizations which do not increase the
  --     code size significantly and which optimize mainly situations
  --     where one or more arguments are compile-time constants.
  --     These optimizations are used always when the compiler is
  --     taught to optimize.
  --   One can inhibit all optimizations by defining __NO_STRING_INLINES.   

  -- Get the machine-dependent optimizations (if any).   
  -- These are generic optimizations which do not add too much inline code.   
  -- Functions with security checks.   
end string_h;
