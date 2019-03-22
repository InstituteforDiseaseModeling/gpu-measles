# setup variables

#################################################################################################
# gencode flags
#SMS ?= 60 61 70 75
#SMS ?= 60 61 70
SMS ?= 60

# Generate SASS code for each SM architecture listed in (SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))

# to echo a variable $(info $$HIGHEST_SM is [${HIGHEST_SM}])

GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
#################################################################################################

#################################################################################################
# link flags
LD_FLAGS=""
#################################################################################################


#################################################################################################
# compile options
CXX_FLAGS=-I/usr/local/include -I../shared -lineinfo -rdc=true -std=c++11 --maxrregcount=63 --ptxas-options=-v
#################################################################################################

measles: *.cu
	nvcc \
		$(GENCODE_FLAGS) \
		$(CXX_FLAGS) \
		-o bin/$@ $^

debug: *.cu
	nvcc \
		$(GENCODE_FLAGS) \
		$(CXX_FLAGS) \
		--debug \
		-o bin/debug $^

clean:
	rm -f bin/*
