# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /users/minesh.mathew/local/bin/cmake

# The command to remove a file.
RM = /users/minesh.mathew/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build

# Include any dependencies generated for this target.
include test/CMakeFiles/storage_test.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/storage_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/storage_test.dir/flags.make

test/CMakeFiles/storage_test.dir/StorageTest.cpp.o: test/CMakeFiles/storage_test.dir/flags.make
test/CMakeFiles/storage_test.dir/StorageTest.cpp.o: ../test/StorageTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/storage_test.dir/StorageTest.cpp.o"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/storage_test.dir/StorageTest.cpp.o -c /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test/StorageTest.cpp

test/CMakeFiles/storage_test.dir/StorageTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/storage_test.dir/StorageTest.cpp.i"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test/StorageTest.cpp > CMakeFiles/storage_test.dir/StorageTest.cpp.i

test/CMakeFiles/storage_test.dir/StorageTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/storage_test.dir/StorageTest.cpp.s"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test/StorageTest.cpp -o CMakeFiles/storage_test.dir/StorageTest.cpp.s

test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.requires:

.PHONY : test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.requires

test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.provides: test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/storage_test.dir/build.make test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.provides.build
.PHONY : test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.provides

test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.provides.build: test/CMakeFiles/storage_test.dir/StorageTest.cpp.o


# Object files for target storage_test
storage_test_OBJECTS = \
"CMakeFiles/storage_test.dir/StorageTest.cpp.o"

# External object files for target storage_test
storage_test_EXTERNAL_OBJECTS =

test/storage_test: test/CMakeFiles/storage_test.dir/StorageTest.cpp.o
test/storage_test: test/CMakeFiles/storage_test.dir/build.make
test/storage_test: libthpp.so
test/storage_test: googletest-release-1.7.0/libgtest.a
test/storage_test: googletest-release-1.7.0/libgtest_main.a
test/storage_test: /users/minesh.mathew/torch/install/lib/libTH.so
test/storage_test: /usr/lib/libopenblas.so
test/storage_test: /usr/lib/liblapack.so
test/storage_test: /users/minesh.mathew/local/lib/libfolly.so
test/storage_test: /usr/lib/x86_64-linux-gnu/libglog.so
test/storage_test: /users/minesh.mathew/local/lib/libthrift.so
test/storage_test: /users/minesh.mathew/local/lib/libthriftcpp2.so
test/storage_test: googletest-release-1.7.0/libgtest.a
test/storage_test: test/CMakeFiles/storage_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable storage_test"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/storage_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/storage_test.dir/build: test/storage_test

.PHONY : test/CMakeFiles/storage_test.dir/build

test/CMakeFiles/storage_test.dir/requires: test/CMakeFiles/storage_test.dir/StorageTest.cpp.o.requires

.PHONY : test/CMakeFiles/storage_test.dir/requires

test/CMakeFiles/storage_test.dir/clean:
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && $(CMAKE_COMMAND) -P CMakeFiles/storage_test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/storage_test.dir/clean

test/CMakeFiles/storage_test.dir/depend:
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test/CMakeFiles/storage_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/storage_test.dir/depend

