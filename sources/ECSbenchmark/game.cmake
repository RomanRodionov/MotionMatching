
set(PROJECT_NAME ECSbenchmark)

include(sdl2.cmake)

include(libs.cmake)


include_library(eastl 3rd_party/eastl)
include_library(ecs ecs)
set(RENDER_ON OFF)
include_library(engine engine)

include_library(imgui 3rd_party/imgui)
include_library(ECSbenchmarkLib ECSbenchmark)

