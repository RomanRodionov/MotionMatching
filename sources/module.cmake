

set(MODULE_SOURCES )
set(MODULE_C_SOURCES )

file(GLOB_RECURSE MODULE_SOURCES ${MODULE_PATH}/*.cpp)
file(GLOB_RECURSE MODULE_C_SOURCES ${MODULE_PATH}/*.c)
include_directories(${MODULE_PATH})

add_library(${MODULE_NAME} OBJECT ${MODULE_SOURCES} ${MODULE_C_SOURCES})
