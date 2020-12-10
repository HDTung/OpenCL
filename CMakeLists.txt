cmake_minimum_required(VERSION 3.5.1)
project(FFT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework OpenCL")

find_package( OpenCV REQUIRED )

include_directories(
        include/
        src/
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd-Party
        ${OpenCV_INCLUDE_DIRS}
)

set(SOURCE_FILES src/main.cpp
        )

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        )
