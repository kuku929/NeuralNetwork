set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/bin/g++-13")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.5.40")
set(CMAKE_CUDA_DEVICE_LINKER "/opt/cuda/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/opt/cuda/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "13.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CUDA_LINKER_DEPFILE_SUPPORTED )

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/opt/cuda")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/opt/cuda")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.5.40")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/opt/cuda")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "86-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/opt/cuda/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/opt/cuda/targets/x86_64-linux/lib/stubs;/opt/cuda/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include/c++;/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include/c++/x86_64-pc-linux-gnu;/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include/c++/backward;/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include;/usr/local/include;/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/opt/cuda/targets/x86_64-linux/lib/stubs;/opt/cuda/targets/x86_64-linux/lib;/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0;/usr/lib;/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_LINKER_LINK "")
set(CMAKE_LINKER_LLD "")
set(CMAKE_CUDA_COMPILER_LINKER "/usr/bin/ld")
set(CMAKE_CUDA_COMPILER_LINKER_ID "GNU")
set(CMAKE_CUDA_COMPILER_LINKER_VERSION 2.42.0)
set(CMAKE_CUDA_COMPILER_LINKER_FRONTEND_VARIANT GNU)
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_MT "")