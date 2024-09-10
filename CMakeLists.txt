cmake_minimum_required(VERSION 3.13)

project(RLMAIFS CXX)

################################################################################
# Set target arch type if empty. Visual studio solution generator provides it.
################################################################################
if(NOT CMAKE_VS_PLATFORM_NAME)
    set(CMAKE_VS_PLATFORM_NAME "x64")
endif()
message("${CMAKE_VS_PLATFORM_NAME} architecture in use")
if(MSVC)
    if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"
        OR "${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86"))
        message(FATAL_ERROR "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
    endif()
endif()

################################################################################
# Global configuration types
################################################################################
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
    CACHE STRING "" FORCE
)


set(PROJECT_NAME RLMAIFS)

option(USE_MPICH "Compile with MPICH" OFF)
option(USE_CUDA "Compile with CUDA" OFF)
option(USE_DOUBLE "Compile with CUDA" OFF)

################################################################################
# Global linker options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_EXE_LINKER_FLAGS "")
    set(CMAKE_MODULE_LINKER_FLAGS "")
    set(CMAKE_SHARED_LINKER_FLAGS "")
    set(CMAKE_STATIC_LINKER_FLAGS "")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS}")
endif()

set(MPICH_INCLUDE_DIR ${MSMPI_INC} CACHE STRING "set MPICH_INCLUDE_DIR")

if (CMAKE_VS_PLATFORM_NAME STREQUAL "x64")
	set(MPI_LIBRARY ${MSMPI_LIB64} CACHE STRING "set MPI_LIBRARY")
else()
	set(MPI_LIBRARY ${MSMPI_LIB32} CACHE STRING "set MPI_LIBRARY")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)   
set(CMAKE_CXX_EXTENSIONS OFF)  
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g" CACHE STRING "Set C++ Compiler Flags" FORCE)
else()
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Set Build Type" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3" CACHE STRING "Set C++ Compiler Flags" FORCE)
endif()
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

file(GLOB_RECURSE SRC
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)
list(REMOVE_ITEM SRC "${CMAKE_SOURCE_DIR}/RLMAIFSMain.cpp")

include_directories(  ${CMAKE_SOURCE_DIR}/include)

include_directories(  ${CMAKE_SOURCE_DIR}/include/RL4FS)

find_package(OpenMP REQUIRED)
if (USE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        link_libraries(${CUDA_LIBRARIES})
    else()
        message(WARNING "cuda not found!")
    endif()
endif()

find_package(_LIGHTGBM)

if (_LIGHTGBM_FOUND)
    message("_LIGHTGBM found")
    link_libraries(${_LIGHTGBM_LIBRARIES})
else()
    message(WARNING "_LIGHTGBM not found, please use -D_LIGHTGBM_INCLUDE_DIRS -D_LIGHTGBM_LIBRARIES !")
    message("_LIGHTGBM_INCLUDE_DIRS: ${_LIGHTGBM_INCLUDE_DIRS} _LIGHTGBM_LIBRARIES: ${_LIGHTGBM_LIBRARIES}")
    link_directories(${_LIGHTGBM_LIBRARIES})
endif()
include_directories(${_LIGHTGBM_INCLUDE_DIRS})



if(OpenMP_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "openmp not found!")
endif()

if (USE_MPICH)
    find_package(MPI)
    if(MPI_FOUND)
        include_directories(${MPI_INCLUDE_PATH})
        link_libraries(${MPI_LIBRARIES})
    else()
        message(WARNING "mpi not found, please use -DMPICH_INCLUDE_DIR -DMPI_LIBRARY !")
        include_directories(${MPICH_INCLUDE_DIR})
        link_directories(${MPI_LIBRARY})
    endif()
endif()



add_library(${PROJECT_NAME} SHARED ${SRC})

add_executable(${PROJECT_NAME}Main ${SRC} ${CMAKE_SOURCE_DIR}/RLMAIFSMain.cpp)

if(MSVC)
    set(ADDITIONAL_LIBRARY_DEPENDENCIES)
    if(USE_MPICH)
        if(NOT MPI_FOUND)
            list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES "msmpi;")    
        endif()
    endif()
    if(NOT _LIGHTGBM_FOUND)
        list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES "lib_lightgbm;")
    endif()
	target_link_libraries( ${PROJECT_NAME}Main PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
   	target_link_libraries( ${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
else()
    if(USE_MPICH)
        set(CMAKE_CXX_COMPILER "mpicxx" CACHE STRING "Set C++ Compiler" FORCE)
        target_link_libraries(${PROJECT_NAME} PRIVATE "mpi")
        target_link_libraries(${PROJECT_NAME}Main PRIVATE "mpi")
        message("link mpi")
    endif()
    if(NOT _LIGHTGBM_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE "_lightgbm")
        target_link_libraries(${PROJECT_NAME}Main PRIVATE "_lightgbm")
        message("link _lightgbm")
    endif()

endif()

if(USE_MPICH)
    message("USE MPICH")
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_MPICH) 
    target_compile_definitions(${PROJECT_NAME}Main PUBLIC USE_MPICH) 
endif()
if(USE_CUDA)
    message("USE CUDA")
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA) 
    target_compile_definitions(${PROJECT_NAME}Main PUBLIC USE_CUDA) 
endif()
if(USE_DOUBLE)
	message("USE DOUBLE")
	target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DOUBLE) 
	target_compile_definitions(${PROJECT_NAME}Main PUBLIC USE_DOUBLE)
endif()


################################################################################
# Post build events
################################################################################
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "debug")
    set(BUILD_DIR_NAME)
    set(OutDir "${CMAKE_CURRENT_BINARY_DIR}/Debug" CACHE STRING "set OutDir" FORCE)
else()
    set(OutDir "${CMAKE_CURRENT_BINARY_DIR}/Release" CACHE STRING "set OutDir" FORCE)
endif()
message("OutDir: ${OutDir}")
if(MSVC)

    add_custom_command(
        TARGET ${PROJECT_NAME}Main
        POST_BUILD
        COMMAND xcopy /Y
        "\"${CMAKE_SOURCE_DIR}/${_LIGHTGBM_INCLUDE_DIRS}/../bin\\*.dll\"" "\"${OutDir}\\\""
    ) 
    
endif()


############################################################
#Install
############################################################

#Header files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
DESTINATION include)

#Library
#Note: may not work on windows

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

install(TARGETS ${PROJECT_NAME}Main
    RUNTIME DESTINATION bin)







