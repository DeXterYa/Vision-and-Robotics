# Install script for directory: /home/dexter/IVR-assignment/ode

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/dexter/IVR-assignment/VRInstall")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "runtime")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so.0.15.2" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so.0.15.2")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so.0.15.2"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/dexter/IVR-assignment/ode-build/libode.so.0.15.2")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so.0.15.2" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so.0.15.2")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so.0.15.2")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/dexter/IVR-assignment/ode-build/libode.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libode.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ode" TYPE FILE FILES
    "/home/dexter/IVR-assignment/ode/include/ode/collision.h"
    "/home/dexter/IVR-assignment/ode/include/ode/collision_space.h"
    "/home/dexter/IVR-assignment/ode/include/ode/collision_trimesh.h"
    "/home/dexter/IVR-assignment/ode/include/ode/common.h"
    "/home/dexter/IVR-assignment/ode/include/ode/compatibility.h"
    "/home/dexter/IVR-assignment/ode/include/ode/contact.h"
    "/home/dexter/IVR-assignment/ode/include/ode/cooperative.h"
    "/home/dexter/IVR-assignment/ode/include/ode/error.h"
    "/home/dexter/IVR-assignment/ode/include/ode/export-dif.h"
    "/home/dexter/IVR-assignment/ode/include/ode/mass.h"
    "/home/dexter/IVR-assignment/ode/include/ode/matrix.h"
    "/home/dexter/IVR-assignment/ode/include/ode/matrix_coop.h"
    "/home/dexter/IVR-assignment/ode/include/ode/memory.h"
    "/home/dexter/IVR-assignment/ode/include/ode/misc.h"
    "/home/dexter/IVR-assignment/ode/include/ode/objects.h"
    "/home/dexter/IVR-assignment/ode/include/ode/ode.h"
    "/home/dexter/IVR-assignment/ode/include/ode/odeconfig.h"
    "/home/dexter/IVR-assignment/ode/include/ode/odecpp.h"
    "/home/dexter/IVR-assignment/ode/include/ode/odecpp_collision.h"
    "/home/dexter/IVR-assignment/ode/include/ode/odeinit.h"
    "/home/dexter/IVR-assignment/ode/include/ode/odemath.h"
    "/home/dexter/IVR-assignment/ode/include/ode/odemath_legacy.h"
    "/home/dexter/IVR-assignment/ode/include/ode/rotation.h"
    "/home/dexter/IVR-assignment/ode/include/ode/threading.h"
    "/home/dexter/IVR-assignment/ode/include/ode/threading_impl.h"
    "/home/dexter/IVR-assignment/ode/include/ode/timer.h"
    "/home/dexter/IVR-assignment/ode-build/include/ode/precision.h"
    "/home/dexter/IVR-assignment/ode-build/include/ode/version.h"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/dexter/IVR-assignment/ode-build/ode.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES "/home/dexter/IVR-assignment/ode-build/ode-config")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2" TYPE FILE FILES "/home/dexter/IVR-assignment/ode-build/ode-config.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2" TYPE FILE FILES "/home/dexter/IVR-assignment/ode-build/ode-config-version.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2/ode-export.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2/ode-export.cmake"
         "/home/dexter/IVR-assignment/ode-build/CMakeFiles/Export/lib/cmake/ode-0.15.2/ode-export.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2/ode-export-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2/ode-export.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2" TYPE FILE FILES "/home/dexter/IVR-assignment/ode-build/CMakeFiles/Export/lib/cmake/ode-0.15.2/ode-export.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ode-0.15.2" TYPE FILE FILES "/home/dexter/IVR-assignment/ode-build/CMakeFiles/Export/lib/cmake/ode-0.15.2/ode-export-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/dexter/IVR-assignment/ode-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
