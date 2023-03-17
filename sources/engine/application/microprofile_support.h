#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <thread>
#include <atomic>

#ifndef MICROPROFILE_SUPPORT

#define MICROPROFILE_SUPPORT
#define MICROPROFILE_GPU_TIMERS_GL 1


#if defined(__APPLE__) || defined(__linux__)
#include <unistd.h>
#endif

#define MICROPROFILE_MAX_FRAME_HISTORY (2<<10)
#define MICROPROFILE_IMPL
#include "microprofile/microprofile.h"

MICROPROFILE_DEFINE(MAIN, "MAIN", "Main", 0xff0000);

#ifdef _WIN32
void usleep(__int64);
#endif
uint32_t g_nQuit = 0;

#endif