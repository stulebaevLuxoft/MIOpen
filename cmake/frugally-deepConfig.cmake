set(frugally-deep_VERSION "0.15.30")

# Create target frugally-deep
add_library(frugally-deep::fdeep INTERFACE IMPORTED)
set_target_properties(frugally-deep::fdeep PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include")
