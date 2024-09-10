#pragma once

#define _CRTDBG_MAP_ALLOC
#include<stdio.h>
#include<stdlib.h>
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(WIN32)
#include<crtdbg.h>
#endif
#ifdef _DEBUG
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#else
#define DBG_NEW new
#endif

