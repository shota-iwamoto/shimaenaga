#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef SHIMAENAGA_BUILD_DLL
    #define SHIMAENAGA_EXPORT __declspec(dllexport)
  #else
    #define SHIMAENAGA_EXPORT __declspec(dllimport)
  #endif
#else
  #define SHIMAENAGA_EXPORT __attribute__((visibility("default")))
#endif
