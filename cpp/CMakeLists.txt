cmake_minimum_required(VERSION 3.16)
project(mc_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# A shared library with a plain C ABI. Not a Python extension module: see
# src/pricing/cpp_backend.py for why that choice matters on this machine.
add_library(mcengine SHARED mc_engine.cpp)

target_compile_options(mcengine PRIVATE -O3 -Wall -Wextra)

# Static-link the compiler runtimes so the DLL is self-contained. Without this
# it imports libwinpthread-1.dll (MinGW's std::thread), which lives in the
# compiler's bin directory -- and Python, started from anywhere else, fails to
# load the DLL with a bare "could not find module" that names the DLL rather
# than the missing dependency. Plain -static covers libgcc, libstdc++ and
# winpthread together.
if(WIN32)
  target_link_options(mcengine PRIVATE -static -Wl,--allow-multiple-definition)
endif()

find_package(Threads REQUIRED)
target_link_libraries(mcengine PRIVATE Threads::Threads)

# Drop the "lib" prefix so the file is mcengine.dll, which is what the Python
# loader looks for first.
set_target_properties(mcengine PROPERTIES PREFIX "")
