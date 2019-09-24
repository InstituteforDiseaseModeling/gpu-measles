#==============================================================================#
# Author: Bismarck Gomes <bismarckgomes@gmail.com>                             #
#                                                                              #
# https://gist.github.com/bismarckjunior/c11872b06d1071f2e0245801c8f72c3b      #
#                                                                              #
#   *                                                                          #
#   L bin          // Executable folder                                        #
#   L build        // Objects (*.o)                                            #
#   L doc          // Documentation                                            #
#   L examples     // Examples                                                 #
#   L include      // Public header files (*.h)                                #
#   L lib          // Library dependencies (*.so, *.dll)                       #
#   L src          // Source files (*.cpp, *.cu) and private header files (*.h)#
#   L test         // Test code files                                          #
#   * makefile                                                                 #
#   * readme.md                                                                #
#                                                                              #
#   # Compile                                                                  #
#   $ make                                                                     #
#                                                                              #
#   # Run                                                                      #
#   $ make run                                                                 #
#                                                                              #
#   # Clean (just files)                                                       #
#   $ make clean                                                               #
#                                                                              #
#   # Clean (files and folders)                                                #
#   $ make clobber                                                             #
#==============================================================================#

# Main variables
SRCDIR := .
#INCDIR :=
INCDIR ?= ../shared /usr/local/include
#BUILDDIR := build
#TARGET := bin/measles

# Debug build flags
ifneq ($(dbg),1)
    TARGET     := bin/measles
	BUILDDIR   := build/release
    BUILD_TYPE := release
else
	TARGET     := bin/debug
	BUILDDIR   := build/debug
    BUILD_TYPE := debug
endif



# Other variables
SOURCES := $(shell find $(SRCDIR) -type f -name "*.cpp")
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))
SOURCESCU := $(shell find $(SRCDIR) -type f -name "*.cu")
OBJECTSCU := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCESCU:.cu=.cu.o))
TARGETDIR = `dirname $(TARGET)`

# Shared
SHARED_BUILDDIR := ../shared/build
SHARED_SRCDIR   := ../shared
SHARED_SOURCES  := $(shell find $(SHARED_SRCDIR) -type f -name "*.cpp")
SHARED_OBJECTS  := $(patsubst $(SHARED_SRCDIR)/%,$(SHARED_BUILDDIR)/%,$(SHARED_SOURCES:.cpp=.o))

# Compilers
HOST_COMPILER := g++
NVCC          := nvcc -ccbin $(HOST_COMPILER)  #/usr/local/cuda/bin/nvcc

# Flags
#NVCCFLAGS     := -m64 -dc  #-dc used cg::grid_group, cg::this_group
NVCCFLAGS     := -lineinfo -rdc=true --maxrregcount=63 # -m64 -dc  #-dc used cg::grid_group, cg::this_group
CCFLAGS       :=
LDFLAGS       :=

# Debug build flags
ifeq ($(dbg),1)
    NVCCFLAGS += --debug
    # NVCCFLAGS += -g -G
endif

# Main flags
ALL_CCFLAGS := $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Includes and libraries
# INCLUDES      := $(addprefix -I ,$(shell find $(SRCDIR) -type d -not -name bin -not -name "build/*"))
INCLUDES      := -I .
LIBRARIES     := 

ifneq ($(INCDIR),)
#     INCLUDES += -I $(INCDIR)
    $(foreach dir,$(INCDIR),$(eval INCLUDES += -I $(dir)))
endif

################################################################################

# Gencode arguments
# SMS ?= 30 35 37 50 52 60 61 70
# SMS ?= 60 61 70
SMS ?= 37
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
    GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

################################################################################

# Target rules
all: build

build: $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -fr $(OBJECTS) $(OBJECTSCU) $(TARGET)

clobber:
	rm -fr $(BUILDDIR) $(TARGETDIR)

$(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu $(SRCDIR)/%.h
	@mkdir -p $(BUILDDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/%.h
	@mkdir -p $(BUILDDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<
#$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(SHARED_BUILDDIR)/%.o : $(SHARED_SRCDIR)/%.cpp $(SHARED_SRCDIR)/%.h
	@mkdir -p $(SHARED_BUILDDIR)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<
#$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(TARGET): $(OBJECTS) $(OBJECTSCU) $(SHARED_OBJECTS)
	@mkdir -p $(TARGETDIR);
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
