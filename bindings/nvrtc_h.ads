pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with System;
with stddef_h;

package nvrtc_h is

  -- * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
  -- *
  -- * NOTICE TO LICENSEE:
  -- *
  -- * This source code and/or documentation ("Licensed Deliverables") are
  -- * subject to NVIDIA intellectual property rights under U.S. and
  -- * international Copyright laws.
  -- *
  -- * These Licensed Deliverables contained herein is PROPRIETARY and
  -- * CONFIDENTIAL to NVIDIA and is being provided under the terms and
  -- * conditions of a form of NVIDIA software license agreement by and
  -- * between NVIDIA and Licensee ("License Agreement") or electronically
  -- * accepted by Licensee.  Notwithstanding any terms or conditions to
  -- * the contrary in the License Agreement, reproduction or disclosure
  -- * of the Licensed Deliverables to any third party without the express
  -- * written consent of NVIDIA is prohibited.
  -- *
  -- * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  -- * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  -- * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
  -- * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  -- * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  -- * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  -- * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  -- * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  -- * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  -- * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  -- * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  -- * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  -- * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  -- * OF THESE LICENSED DELIVERABLES.
  -- *
  -- * U.S. Government End Users.  These Licensed Deliverables are a
  -- * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  -- * 1995), consisting of "commercial computer software" and "commercial
  -- * computer software documentation" as such terms are used in 48
  -- * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
  -- * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  -- * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  -- * U.S. Government End Users acquire the Licensed Deliverables with
  -- * only those rights set forth herein.
  -- *
  -- * Any use of the Licensed Deliverables in individual and commercial
  -- * software must include, in the user documentation and internal
  -- * comments to the code, the above Disclaimer and U.S. Government End
  -- * Users Notice.
  --  

  --*********************************************************************** 
  --*
  -- *
  -- * \defgroup error Error Handling
  -- *
  -- * NVRTC defines the following enumeration type and function for API call
  -- * error handling.
  -- *
  -- *************************************************************************** 

  --*
  -- * \ingroup error
  -- * \brief   The enumerated type nvrtcResult defines API call result codes.
  -- *          NVRTC API functions return nvrtcResult to indicate the call
  -- *          result.
  --  

   type nvrtcResult is 
     (NVRTC_SUCCESS,
      NVRTC_ERROR_OUT_OF_MEMORY,
      NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
      NVRTC_ERROR_INVALID_INPUT,
      NVRTC_ERROR_INVALID_PROGRAM,
      NVRTC_ERROR_INVALID_OPTION,
      NVRTC_ERROR_COMPILATION,
      NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
      NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
      NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
      NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
      NVRTC_ERROR_INTERNAL_ERROR);
   pragma Convention (C, nvrtcResult);  -- /usr/local/cuda-8.0/include/nvrtc.h:89

  --*
  -- * \ingroup error
  -- * \brief   nvrtcGetErrorString is a helper function that returns a string
  -- *          describing the given nvrtcResult code, e.g., NVRTC_SUCCESS to
  -- *          \c "NVRTC_SUCCESS".
  -- *          For unrecognized enumeration values, it returns
  -- *          \c "NVRTC_ERROR unknown".
  -- *
  -- * \param   [in] result CUDA Runtime Compilation API result code.
  -- * \return  Message string for the given #nvrtcResult code.
  --  

   function nvrtcGetErrorString (result : nvrtcResult) return Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvrtc.h:103
   pragma Import (C, nvrtcGetErrorString, "nvrtcGetErrorString");

  --*********************************************************************** 
  --*
  -- *
  -- * \defgroup query General Information Query
  -- *
  -- * NVRTC defines the following function for general information query.
  -- *
  -- *************************************************************************** 

  --*
  -- * \ingroup query
  -- * \brief   nvrtcVersion sets the output parameters \p major and \p minor
  -- *          with the CUDA Runtime Compilation version number.
  -- *
  -- * \param   [out] major CUDA Runtime Compilation major version number.
  -- * \param   [out] minor CUDA Runtime Compilation minor version number.
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
  -- *
  --  

   function nvrtcVersion (major : access int; minor : access int) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:127
   pragma Import (C, nvrtcVersion, "nvrtcVersion");

  --*********************************************************************** 
  --*
  -- *
  -- * \defgroup compilation Compilation
  -- *
  -- * NVRTC defines the following type and functions for actual compilation.
  -- *
  -- *************************************************************************** 

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcProgram is the unit of compilation, and an opaque handle for
  -- *          a program.
  -- *
  -- * To compile a CUDA program string, an instance of nvrtcProgram must be
  -- * created first with ::nvrtcCreateProgram, then compiled with
  -- * ::nvrtcCompileProgram.
  --  

   --  skipped empty struct u_nvrtcProgram

   type nvrtcProgram is new System.Address;  -- /usr/local/cuda-8.0/include/nvrtc.h:148

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcCreateProgram creates an instance of nvrtcProgram with the
  -- *          given input parameters, and sets the output parameter \p prog with
  -- *          it.
  -- *
  -- * \param   [out] prog         CUDA Runtime Compilation program.
  -- * \param   [in]  src          CUDA program source.
  -- * \param   [in]  name         CUDA program name.\n
  -- *                             \p name can be \c NULL; \c "default_program" is
  -- *                             used when \p name is \c NULL.
  -- * \param   [in]  numHeaders   Number of headers used.\n
  -- *                             \p numHeaders must be greater than or equal to 0.
  -- * \param   [in]  headers      Sources of the headers.\n
  -- *                             \p headers can be \c NULL when \p numHeaders is
  -- *                             0.
  -- * \param   [in]  includeNames Name of each header by which they can be
  -- *                             included in the CUDA program source.\n
  -- *                             \p includeNames can be \c NULL when \p numHeaders
  -- *                             is 0.
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_PROGRAM_CREATION_FAILURE \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
  -- *
  -- * \see     ::nvrtcDestroyProgram
  --  

   function nvrtcCreateProgram
     (prog : System.Address;
      src : Interfaces.C.Strings.chars_ptr;
      name : Interfaces.C.Strings.chars_ptr;
      numHeaders : int;
      headers : System.Address;
      includeNames : System.Address) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:180
   pragma Import (C, nvrtcCreateProgram, "nvrtcCreateProgram");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcDestroyProgram destroys the given program.
  -- *
  -- * \param    [in] prog CUDA Runtime Compilation program.
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
  -- *
  -- * \see     ::nvrtcCreateProgram
  --  

   function nvrtcDestroyProgram (prog : System.Address) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:199
   pragma Import (C, nvrtcDestroyProgram, "nvrtcDestroyProgram");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcCompileProgram compiles the given program.
  -- *
  -- * It supports compile options listed in \ref options.
  --  

   function nvrtcCompileProgram
     (prog : nvrtcProgram;
      numOptions : int;
      options : System.Address) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:208
   pragma Import (C, nvrtcCompileProgram, "nvrtcCompileProgram");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcGetPTXSize sets \p ptxSizeRet with the size of the PTX
  -- *          generated by the previous compilation of \p prog (including the
  -- *          trailing \c NULL).
  -- *
  -- * \param   [in]  prog       CUDA Runtime Compilation program.
  -- * \param   [out] ptxSizeRet Size of the generated PTX (including the trailing
  -- *                           \c NULL).
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
  -- *
  -- * \see     ::nvrtcGetPTX
  --  

   function nvrtcGetPTXSize (prog : nvrtcProgram; ptxSizeRet : access stddef_h.size_t) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:228
   pragma Import (C, nvrtcGetPTXSize, "nvrtcGetPTXSize");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcGetPTX stores the PTX generated by the previous compilation
  -- *          of \p prog in the memory pointed by \p ptx.
  -- *
  -- * \param   [in]  prog CUDA Runtime Compilation program.
  -- * \param   [out] ptx  Compiled result.
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
  -- *
  -- * \see     ::nvrtcGetPTXSize
  --  

   function nvrtcGetPTX (prog : nvrtcProgram; ptx : Interfaces.C.Strings.chars_ptr) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:245
   pragma Import (C, nvrtcGetPTX, "nvrtcGetPTX");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcGetProgramLogSize sets \p logSizeRet with the size of the
  -- *          log generated by the previous compilation of \p prog (including the
  -- *          trailing \c NULL).
  -- *
  -- * Note that compilation log may be generated with warnings and informative
  -- * messages, even when the compilation of \p prog succeeds.
  -- *
  -- * \param   [in]  prog       CUDA Runtime Compilation program.
  -- * \param   [out] logSizeRet Size of the compilation log
  -- *                           (including the trailing \c NULL).
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
  -- *
  -- * \see     ::nvrtcGetProgramLog
  --  

   function nvrtcGetProgramLogSize (prog : nvrtcProgram; logSizeRet : access stddef_h.size_t) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:267
   pragma Import (C, nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcGetProgramLog stores the log generated by the previous
  -- *          compilation of \p prog in the memory pointed by \p log.
  -- *
  -- * \param   [in]  prog CUDA Runtime Compilation program.
  -- * \param   [out] log  Compilation log.
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
  -- *
  -- * \see     ::nvrtcGetProgramLogSize
  --  

   function nvrtcGetProgramLog (prog : nvrtcProgram; log : Interfaces.C.Strings.chars_ptr) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:284
   pragma Import (C, nvrtcGetProgramLog, "nvrtcGetProgramLog");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcAddNameExpression notes the given name expression
  -- *          denoting a __global__ function or function template
  -- *          instantiation.
  -- *
  -- * The identical name expression string must be provided on a subsequent
  -- * call to nvrtcGetLoweredName to extract the lowered name.
  -- * \param   [in]  prog CUDA Runtime Compilation program.
  -- * \param   [in] name_expression constant expression denoting a __global__
  -- *               function or function template instantiation.
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION \endlink
  -- *
  -- * \see     ::nvrtcGetLoweredName
  --  

   function nvrtcAddNameExpression (prog : nvrtcProgram; name_expression : Interfaces.C.Strings.chars_ptr) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:304
   pragma Import (C, nvrtcAddNameExpression, "nvrtcAddNameExpression");

  --*
  -- * \ingroup compilation
  -- * \brief   nvrtcGetLoweredName extracts the lowered (mangled) name
  -- *          for a __global__ function or function template instantiation,
  -- *          and updates *lowered_name to point to it. The memory containing
  -- *          the name is released when the NVRTC program is destroyed by 
  -- *          nvrtcDestroyProgram.
  -- *          The identical name expression must have been previously
  -- *          provided to nvrtcAddNameExpression.
  -- *
  -- * \param   [in]  prog CUDA Runtime Compilation program.
  -- * \param   [in] name_expression constant expression denoting a __global__
  -- *               function or function template instantiation.
  -- * \param   [out] lowered_name initialized by the function to point to a
  -- *               C string containing the lowered (mangled)
  -- *               name corresponding to the provided name expression.
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID \endlink
  -- *
  -- * \see     ::nvrtcAddNameExpression
  --  

   function nvrtcGetLoweredName
     (prog : nvrtcProgram;
      name_expression : Interfaces.C.Strings.chars_ptr;
      lowered_name : System.Address) return nvrtcResult;  -- /usr/local/cuda-8.0/include/nvrtc.h:330
   pragma Import (C, nvrtcGetLoweredName, "nvrtcGetLoweredName");

  --*
  -- * \defgroup options Supported Compile Options
  -- *
  -- * NVRTC supports the compile options below.
  -- * Option names with two preceding dashs (\c --) are long option names and
  -- * option names with one preceding dash (\c -) are short option names.
  -- * Short option names can be used instead of long option names.
  -- * When a compile option takes an argument, an assignment operator (\c =)
  -- * is used to separate the compile option argument from the compile option
  -- * name, e.g., \c "--gpu-architecture=compute_20".
  -- * Alternatively, the compile option name and the argument can be specified in
  -- * separate strings without an assignment operator, .e.g,
  -- * \c "--gpu-architecturend" \c "compute_20".
  -- * Single-character short option names, such as \c -D, \c -U, and \c -I, do
  -- * not require an assignment operator, and the compile option name and the
  -- * argument can be present in the same string with or without spaces between
  -- * them.
  -- * For instance, \c "-D=<def>", \c "-D<def>", and \c "-D <def>" are all
  -- * supported.
  -- *
  -- * The valid compiler options are:
  -- *
  -- *   - Compilation targets
  -- *     - \c --gpu-architecture=\<arch\> (\c -arch)\n
  -- *       Specify the name of the class of GPU architectures for which the
  -- *       input must be compiled.\n
  -- *       - Valid <c>\<arch\></c>s:
  -- *         - \c compute_20
  -- *         - \c compute_30
  -- *         - \c compute_35
  -- *         - \c compute_50
  -- *         - \c compute_52
  -- *         - \c compute_53
  -- *       - Default: \c compute_20
  -- *   - Separate compilation / whole-program compilation
  -- *     - \c --device-c (\c -dc)\n
  -- *       Generate relocatable code that can be linked with other relocatable
  -- *       device code.  It is equivalent to --relocatable-device-code=true.
  -- *     - \c --device-w (\c -dw)\n
  -- *       Generate non-relocatable code.  It is equivalent to
  -- *       \c --relocatable-device-code=false.
  -- *     - \c --relocatable-device-code={true|false} (\c -rdc)\n
  -- *       Enable (disable) the generation of relocatable device code.
  -- *       - Default: \c false
  -- *   - Debugging support
  -- *     - \c --device-debug (\c -G)\n
  -- *       Generate debug information.
  -- *     - \c --generate-line-info (\c -lineinfo)\n
  -- *       Generate line-number information.
  -- *   - Code generation
  -- *     - \c --maxrregcount=\<N\> (\c -maxrregcount)\n
  -- *       Specify the maximum amount of registers that GPU functions can use.
  -- *       Until a function-specific limit, a higher value will generally
  -- *       increase the performance of individual GPU threads that execute this
  -- *       function.  However, because thread registers are allocated from a
  -- *       global register pool on each GPU, a higher value of this option will
  -- *       also reduce the maximum thread block size, thereby reducing the amount
  -- *       of thread parallelism.  Hence, a good maxrregcount value is the result
  -- *       of a trade-off.  If this option is not specified, then no maximum is
  -- *       assumed.  Value less than the minimum registers required by ABI will
  -- *       be bumped up by the compiler to ABI minimum limit.
  -- *     - \c --ftz={true|false} (\c -ftz)\n
  -- *       When performing single-precision floating-point operations, flush
  -- *       denormal values to zero or preserve denormal values.
  -- *       \c --use_fast_math implies \c --ftz=true.
  -- *       - Default: \c false
  -- *     - \c --prec-sqrt={true|false} (\c -prec-sqrt)\n
  -- *       For single-precision floating-point square root, use IEEE
  -- *       round-to-nearest mode or use a faster approximation.
  -- *       \c --use_fast_math implies \c --prec-sqrt=false.
  -- *       - Default: \c true
  -- *     - \c --prec-div={true|false} (\c -prec-div)\n
  -- *       For single-precision floating-point division and reciprocals, use IEEE
  -- *       round-to-nearest mode or use a faster approximation.
  -- *       \c --use_fast_math implies \c --prec-div=false.
  -- *       - Default: \c true
  -- *     - \c --fmad={true|false} (\c -fmad)\n
  -- *       Enables (disables) the contraction of floating-point multiplies and
  -- *       adds/subtracts into floating-point multiply-add operations (FMAD,
  -- *       FFMA, or DFMA).  \c --use_fast_math implies \c --fmad=true.
  -- *       - Default: \c true
  -- *     - \c --use_fast_math (\c -use_fast_math)\n
  -- *       Make use of fast math operations.
  -- *       \c --use_fast_math implies \c --ftz=true \c --prec-div=false
  -- *       \c --prec-sqrt=false \c --fmad=true.
  -- *   - Preprocessing
  -- *     - \c --define-macro=\<def\> (\c -D)\n
  -- *       \c \<def\> can be either \c \<name\> or \c \<name=definitions\>.
  -- *       - \c \<name\> \n
  -- *         Predefine \c \<name\> as a macro with definition \c 1.
  -- *       - \c \<name\>=\<definition\> \n
  -- *         The contents of \c \<definition\> are tokenized and preprocessed
  -- *         as if they appeared during translation phase three in a \c \#define
  -- *         directive.  In particular, the definition will be truncated by
  -- *         embedded new line characters.
  -- *     - \c --undefine-macro=\<def\> (\c -U)\n
  -- *       Cancel any previous definition of \c \<def\>.
  -- *     - \c --include-path=\<dir\> (\c -I)\n
  -- *       Add the directory \c \<dir\> to the list of directories to be
  -- *       searched for headers.  These paths are searched after the list of
  -- *       headers given to ::nvrtcCreateProgram.
  -- *     - \c --pre-include=\<header\> (\c -include)\n
  -- *       Preinclude \c \<header\> during preprocessing.
  -- *   - Language Dialect
  -- *     - \c --std=c++11 (\c -std=c++11)\n
  -- *       Set language dialect to C++11.
  -- *     - \c --builtin-move-forward={true|false} (\c -builtin-move-forward)\n
  -- *       Provide builtin definitions of \c std::move and \c std::forward,
  -- *       when C++11 language dialect is selected.
  -- *       - Default: \c true
  -- *     - \c --builtin-initializer-list={true|false}
  -- *       (\c -builtin-initializer-list)\n
  -- *       Provide builtin definitions of \c std::initializer_list class and
  -- *       member functions when C++11 language dialect is selected.
  -- *       - Default: \c true
  -- *   - Misc.
  -- *     - \c --disable-warnings (\c -w)\n
  -- *       Inhibit all warning messages.
  -- *     - \c --restrict (\c -restrict)\n
  -- *       Programmer assertion that all kernel pointer parameters are restrict
  -- *       pointers.
  -- *     - \c --device-as-default-execution-space
  -- *       (\c -default-device)\n
  -- *       Treat entities with no execution space annotation as \c __device__
  -- *       entities.
  -- *
  -- * \param   [in] prog       CUDA Runtime Compilation program.
  -- * \param   [in] numOptions Number of compiler options passed.
  -- * \param   [in] options    Compiler options in the form of C string array.\n
  -- *                          \p options can be \c NULL when \p numOptions is 0.
  -- *
  -- * \return
  -- *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_INVALID_OPTION \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_COMPILATION \endlink
  -- *   - \link #nvrtcResult NVRTC_ERROR_BUILTIN_OPERATION_FAILURE \endlink
  --  

  -- The utility function 'nvrtcGetTypeName' is not available by default. Define
  --   the macro 'NVRTC_GET_TYPE_NAME' to a non-zero value to make it available.
  -- 

  --*********************************************************************** 
  --*
  -- *
  -- * \defgroup hosthelper Host Helper
  -- *
  -- * NVRTC defines the following functions for easier interaction with host code.
  -- *
  -- *************************************************************************** 

  --*
  -- * \ingroup hosthelper
  -- * \brief   nvrtcGetTypeName stores the source level name of the template type argument
  -- *          T in the given std::string location.
  -- *
  -- * This function is only provided when the macro NVRTC_GET_TYPE_NAME is
  -- * defined with a non-zero value. It uses abi::__cxa_demangle or UnDecorateSymbolName
  -- * function calls to extract the type name, when using gcc/clang or cl.exe compilers,
  -- * respectively. If the name extraction fails, it will return NVRTC_INTERNAL_ERROR,
  -- * otherwise *result is initialized with the extracted name.
  -- * 
  -- * \param   [in] result: pointer to std::string in which to store the type name.
  -- * \return
  -- *  - \link #nvrtcResult NVRTC_SUCCESS \endlink
  -- *  - \link #nvrtcResult NVRTC_ERROR_INTERNAL_ERROR \endlink
  -- *
  --  

end nvrtc_h;
