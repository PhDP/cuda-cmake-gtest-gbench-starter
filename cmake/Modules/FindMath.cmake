# - Find MATH library
# Find the native MATH includes and library
# This module defines
# MATH_INCLUDE_DIR, where to find hdf5.h, etc.
# MATH_LIBRARIES, libraries to link against to use MATH.
# MATH_FOUND, If false, do not try to use MATH.
# also defined, but not for general use are
# MATH_LIBRARY, where to find the MATH library.

find_path(MATH_INCLUDE_DIR math.h)
set(MATH_NAMES ${MATH_NAMES} m)
find_library(MATH_LIBRARY NAMES ${MATH_NAMES})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MATH DEFAULT_MSG MATH_LIBRARY MATH_INCLUDE_DIR)

if(MATH_FOUND)
 set(MATH_LIBRARIES ${MATH_LIBRARY})
endif(MATH_FOUND)

