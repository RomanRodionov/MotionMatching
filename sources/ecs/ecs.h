#pragma once
#include <stdint.h>
typedef uint32_t uint;
#include "ecs_core.h"
#include "ecs_stage.h"
#include "ecs_event.h"
#include "ecs_tag.h"
#define SYSTEM(...)static inline void 
#define QUERY(...)
#define EVENT(...)static inline void 