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
include test/CMakeFiles/tensor_serialization_test.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/tensor_serialization_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/tensor_serialization_test.dir/flags.make

test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o: test/CMakeFiles/tensor_serialization_test.dir/flags.make
test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o: ../test/TensorSerializationTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o -c /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test/TensorSerializationTest.cpp

test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.i"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test/TensorSerializationTest.cpp > CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.i

test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.s"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test/TensorSerializationTest.cpp -o CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.s

test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.requires:

.PHONY : test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.requires

test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.provides: test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/tensor_serialization_test.dir/build.make test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.provides.build
.PHONY : test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.provides

test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.provides.build: test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o


# Object files for target tensor_serialization_test
tensor_serialization_test_OBJECTS = \
"CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o"

# External object files for target tensor_serialization_test
tensor_serialization_test_EXTERNAL_OBJECTS =

test/tensor_serialization_test: test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o
test/tensor_serialization_test: test/CMakeFiles/tensor_serialization_test.dir/build.make
test/tensor_serialization_test: libthpp.so
test/tensor_serialization_test: googletest-release-1.7.0/libgtest.a
test/tensor_serialization_test: googletest-release-1.7.0/libgtest_main.a
test/tensor_serialization_test: /users/minesh.mathew/torch/install/lib/libTH.so
test/tensor_serialization_test: /usr/lib/libopenblas.so
test/tensor_serialization_test: /usr/lib/liblapack.so
test/tensor_serialization_test: /users/minesh.mathew/local/lib/libfolly.so
test/tensor_serialization_test: /usr/lib/x86_64-linux-gnu/libglog.so
test/tensor_serialization_test: /users/minesh.mathew/local/lib/libthrift.so
test/tensor_serialization_test: /users/minesh.mathew/local/lib/libthriftcpp2.so
test/tensor_serialization_test: googletest-release-1.7.0/libgtest.a
test/tensor_serialization_test: test/CMakeFiles/tensor_serialization_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tensor_serialization_test"
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tensor_serialization_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/tensor_serialization_test.dir/build: test/tensor_serialization_test

.PHONY : test/CMakeFiles/tensor_serialization_test.dir/build

test/CMakeFiles/tensor_serialization_test.dir/requires: test/CMakeFiles/tensor_serialization_test.dir/TensorSerializationTest.cpp.o.requires

.PHONY : test/CMakeFiles/tensor_serialization_test.dir/requires

test/CMakeFiles/tensor_serialization_test.dir/clean:
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test && $(CMAKE_COMMAND) -P CMakeFiles/tensor_serialization_test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/tensor_serialization_test.dir/clean

test/CMakeFiles/tensor_serialization_test.dir/depend:
	cd /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/test /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test /OCRData/minesh.mathew/cnn_rnn/thpp-1.0/thpp/build/test/CMakeFiles/tensor_serialization_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/tensor_serialization_test.dir/depend

