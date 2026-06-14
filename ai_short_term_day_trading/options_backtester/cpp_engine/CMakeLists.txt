cmake_minimum_required(VERSION 3.12)
project(OptionsReplayEngine VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
# In a real environment, user needs `pip install pybind11` and perhaps specify python executable
# E.g.: cmake -DPYTHON_EXECUTABLE=$(which python) ..
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11 installed via pip
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${_pybind11_path}")
find_package(pybind11 CONFIG REQUIRED)

# Add source files
set(SOURCES
    src/orderbook_replay.cpp
    src/bindings.cpp
)

# Create pybind11 module
pybind11_add_module(options_replay ${SOURCES})

if(WIN32 AND MINGW)
    set_target_properties(options_replay PROPERTIES LINK_FLAGS "-static -static-libgcc -static-libstdc++")
endif()

# Include directories
target_include_directories(options_replay PRIVATE src)
