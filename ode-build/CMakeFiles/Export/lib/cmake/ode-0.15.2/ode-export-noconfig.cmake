#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ODE::ODE" for configuration ""
set_property(TARGET ODE::ODE APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ODE::ODE PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "-lpthread"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libode.so.0.15.2"
  IMPORTED_SONAME_NOCONFIG "libode.so.0.15.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS ODE::ODE )
list(APPEND _IMPORT_CHECK_FILES_FOR_ODE::ODE "${_IMPORT_PREFIX}/lib/libode.so.0.15.2" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
