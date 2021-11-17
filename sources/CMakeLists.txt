cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME Application)

if(WIN32)
    set(CMAKE_CXX_COMPILER "C:/Program Files/LLVM/bin/clang++.exe")
    set(CMAKE_C_COMPILER "C:/Program Files/LLVM/bin/clang.exe")
    set(ADDITIONAL_LIBS winmm.lib Version.lib Setupapi.lib)
else()
    set(ADDITIONAL_LIBS -ldl)
endif()

project(${PROJECT_NAME})


set(ROOT ${CMAKE_SOURCE_DIR}/..)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-m64 -Wall -Wextra -Wno-pragma-pack -Wno-deprecated-declarations")


if(BUILD_TYPE STREQUAL "dbg")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    if (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
    set(CMAKE_BUILD_TYPE  "Release")
endif()

if(BUILD_TYPE STREQUAL "rel")
    add_definitions(-DRELEASE)
endif()

set(is3DApplication -1)
list(FIND DEPENDENCIES "3rd_party/3d_application" is3DApplication)

if(${is3DApplication} GREATER -1)
    if(WIN32)
        set(ADDITIONAL_LIBS ${ADDITIONAL_LIBS} SDL2 assimp)
        set (BuildDir "${ROOT}/Builds")
        add_library(SDL2 STATIC IMPORTED)
        set_target_properties(SDL2 PROPERTIES
            IMPORTED_LOCATION_RELEASE "${BuildDir}/libSDL2.a"
            IMPORTED_LOCATION_DEBUG "${BuildDir}/libSDL2.a"
            LINKER_LANGUAGE CXX)
        add_library(assimp STATIC IMPORTED)
        set_target_properties(assimp PROPERTIES
                IMPORTED_LOCATION_DEBUG "${BuildDir}/assimp-vc142-mt.lib"
                IMPORTED_LOCATION_RELEASE "${BuildDir}/assimp-vc142-mt.lib"
                LINKER_LANGUAGE CXX)
    else()
        find_package(SDL2 REQUIRED)
        include_directories(${SDL2_INCLUDE_DIRS})
        find_package(assimp REQUIRED)
    endif()
endif()



foreach(DEPENDENCY ${DEPENDENCIES})
  include_directories(${CMAKE_SOURCE_DIR}/${DEPENDENCY})
endforeach()

foreach(DEPENDENCY ${DEPENDENCIES})
  set(MODULE_PATH ${CMAKE_SOURCE_DIR}/${DEPENDENCY})
  get_filename_component(MODULE_NAME ${MODULE_PATH} NAME)
  set(MODULE_LIST ${MODULE_LIST} ${MODULE_NAME})
  add_subdirectory(${MODULE_PATH})
endforeach()


add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${MODULE_LIST} ${ADDITIONAL_LIBS})