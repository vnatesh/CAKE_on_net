.PHONY: all compile clean

ifneq ($(strip $(BLIS_INSTALL_PATH)),)
BLIS_INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
endif

# g++ cake_sgemm_test.cpp -I/usr/local/include/blis -I/tmp/CAKE_on_CPU/include -L/tmp/CAKE_on_CPU  -lcake -o testing

CAKE_INC_PATH  := $(CAKE_HOME)/include
C2_INC_PATH	   := $(PWD)/include
TEST_OBJ_PATH  := .

# Use the "framework" CFLAGS for the configuration family.
CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS_tmp        += -I$(C2_INC_PATH)
CFLAGS_tmp        += -I$(CAKE_INC_PATH)
CFLAGS_tmp		  += -I$(BLIS_INC_PATH)
CFLAGS 	:= $(filter-out -std=c99, $(CFLAGS_tmp))

# cake shared library
LIBCAKE      := -L$(CAKE_HOME) -lcake
CAKE_SRC := $(CAKE_HOME)/src


# --- Targets/rules ------------------------------------------------------------

all: cake

cake: $(wildcard *.h) $(wildcard *.c)
	mpic++ -g -O3 -Wall $(CFLAGS) src/cake_sgemm_c2.cpp src/pack_c2.cpp \
	src/block_sizing_c2.cpp src/cake_thr.cpp src/helper.cpp cake_sgemm_test.cpp $(LIBCAKE) -fopenmp -pthread \
	-DUSE_CAKE -o cake_sgemm_test

mkl: $(wildcard *.h) $(wildcard *.c)
	mpic++ -fopenmp -m64 -I${MKLROOT}/include -I${CAKE_HOME} -I${C2_INC_PATH} \
	-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  -lmkl_intel_lp64 -lmkl_core \
	-lmkl_gnu_thread -lpthread -lm -ldl src/cake_sgemm_c2.cpp src/pack_c2.cpp \
	src/block_sizing_c2.cpp src/mkl_thr.cpp src/util.cpp src/helper.cpp \
	cake_sgemm_test.cpp -DUSE_MKL -o cake_sgemm_test


clean:
	rm -rf *.o *.so

