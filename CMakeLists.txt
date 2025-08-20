cmake_minimum_required(VERSION 3.20)
# Project name = folder name by default; you can override it:
project(GenericOpenCVProject CXX)
set(CMAKE_CXX_STANDARD 17)

# --- OpenCV location ---
# Option A (recommended): set an environment variable OPENCV_DIR once on your system
#  e.g., OPENCV_DIR=C:/opencv/build
# Then this line will pick it up automatically:
if (NOT DEFINED OpenCV_DIR AND DEFINED ENV{OPENCV_DIR})
  set(OpenCV_DIR "$ENV{OPENCV_DIR}")
endif()

# Option B: hardcode if you prefer (uncomment and adjust):
# set(OpenCV_DIR "C:/opencv/build")

find_package(OpenCV REQUIRED)

# --- Sources: build ALL .cpp files in this folder automatically ---
file(GLOB SRC_FILES "*.cpp")
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Optional (Windows TTS libs if you use SAPI in your code)
if (WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE ole32.lib sapi.lib)
endif()

# Run from source folder so assets like yolov8n.onnx are found easily
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Convenience: copy OpenCV DLLs next to the built exe after build (Windows prebuilt layout)
if (WIN32 AND EXISTS "${OpenCV_DIR}/x64")
  # vc16 is common for VS2019/2022 prebuilt OpenCV; change to vc17 if your package uses it
  if (EXISTS "${OpenCV_DIR}/x64/vc16/bin")
    set(_opencv_bin "${OpenCV_DIR}/x64/vc16/bin")
  elseif (EXISTS "${OpenCV_DIR}/x64/vc17/bin")
    set(_opencv_bin "${OpenCV_DIR}/x64/vc17/bin")
  endif()

  if (DEFINED _opencv_bin)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${_opencv_bin}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
      COMMENT "Copying OpenCV DLLs to output dir")
  endif()
endif()
