
set(MODULE_NAME ecs)
set(MODULE_PATH ${SRC_ROOT}/${MODULE_NAME})

include_directories(${SRC_ROOT}/3rd_party/include)
include_directories(${SRC_ROOT}/3rd_party/imgui)
include_directories(${SRC_ROOT}/3rd_party/eastl/include)
include_directories(${SRC_ROOT}/engine)
include(../module.cmake)
