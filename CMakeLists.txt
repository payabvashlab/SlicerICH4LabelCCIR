cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerICH4LabelCCIR)

set(EXTENSION_HOMEPAGE "https://github.com/KitwareMedical/SlicerNNUnet")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Anh Tuan Tran, Sam Payabvash")
set(EXTENSION_DESCRIPTION "Single-volume or batch four-label NCCT segmentation with ORBIT-RSNA followed by a configurable component-conditioned IVH refiner.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "PyTorch")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(ICH4LabelCCIR)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
