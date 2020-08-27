include $(RTE_SDK)/mk/rte.vars.mk

SRC_DIR := $(shell pwd)/src
APP_DIR := $(shell pwd)/apps
LIB_DIR := $(shell pwd)/lib
OBJ_DIR := $(shell pwd)/obj
INC = -I$(shell pwd)/lib -I$(APP_DIR)/lib

CUDA ?= /usr/local/cuda-10.1
CUDA_LIBPATH := -L$(CUDA)/lib64 -L$(CUDA)/lib -L/usr/lib64/nvidia -L/usr/lib/nvidia
CUDA_INC := -I$(CUDA)/include 

SMV = 30
PKGCONF = pkg-config

ifeq ($(shell pkg-config --exists libdpdk && echo 0),0)

PC_FILE := $(shell $(PKGCONF) --path libdpdk 2>/dev/null)
CFLAGS = -O3 $(shell $(PKGCONF) --cflags libdpdk)
LDFLAGS_SHARED = $(shell $(PKGCONF) --libs libdpdk)

ifndef BUILD

sixteen := x x x x x x x x x x x x x x x x
MAX := $(foreach x, $(sixteen), $(sixteen))

T := $(shell $(MAKE) -nrRf $(firstword $(MAKEFILE_LIST)) $(MAKECMDGOALS) \
	BUILD="COUNTTHIS" | grep -c "COUNTTHIS")

N := $(wordlist 1, $T, $(MAX))

counter = $$(($(T)-$(words $N) + 1))$(eval N := $(wordlist 2,$(words $N),$N))

BUILD = @echo $(counter) of $(T)
endif


all: dpdkIO dpdkGPU dpdkGPUONE dpdkGPUNF
	@echo done

#io compile
dpdkIO: util.o dpdkio.o mainio.o
	@g++ -o $@ -m64 -pthread \
	$(CFLAGS) $^ $(CUDA_LIBPATH) $(CUDA_INC) $(INC) $(LDFLAGS_SHARED) 
	$(BUILD) dpdkIO
	@mv *.o $(OBJ_DIR)

dpdkio.o: $(SRC_DIR)/dpdkio.c
	@gcc -w -m64 -pthread $(CFLAGS) $(LDFLAGS_SHARED) -o $@ -c $^ $(INC) \
	$(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) dpdkio

mainio.o: ./main/mainio.c
	@gcc -w -o $@ -c $^ $(INC)  $(CFLAGS) $(LDFLAGS) $(LDFLAGS_SHARED) -ldl -pthread -lnuma
	$(BUILD) mainio

#gpu compile
dpdkGPU: master.o gpu_forwarder.o dpdk_gpu.o util.o maingpu.o
	@g++ -g -o $@ -m64 -pthread \
	$(CFLAGS) $^ $(CUDA_LIBPATH) $(CUDA_INC) $(INC) $(LDFLAGS_SHARED) -lcudart 
	$(BUILD) dpdkGPU
	@mv *.o $(OBJ_DIR)

dpdk_gpu.o: $(SRC_DIR)/dpdk_gpu.c
	@gcc -g -w -m64 -pthread $(CFLAGS) $(LDFLAGS_SHARED) -o $@ -c $^ $(INC) 
	$(BUILD) dpdk_gpu

master.o: $(SRC_DIR)/master.c
	@gcc -g -w -m64 -pthread $(CFLAGS) $(LDFLAGS_SHARED) -o $@ -c $^ $(INC) \
	$(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) master

#gpuone compile
dpdkGPUONE: gpu_forwarder.o dpdk_gpuone.o util.o maingpu.o
	@g++ -g -o $@ -m64 -pthread \
	$(CFLAGS) $^ $(CUDA_LIBPATH) $(CUDA_INC) $(INC) $(LDFLAGS_SHARED) -lcudart 
	$(BUILD) dpdkGPU
	@mv *.o $(OBJ_DIR)

dpdk_gpuone.o: $(SRC_DIR)/dpdk_gpuone.c
	@gcc -g -w -m64 -pthread $(CFLAGS) $(LDFLAGS_SHARED) -o $@ -c $^ $(INC) 
#	$(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) dpdk_gpu

#gpu with nf compile
dpdkGPUNF: router.o nids.o ipsec.o master.o gpu_forwarder.o dpdk_gpunf.o util.o maingpunf.o
	@g++ -g -o $@ -m64 -pthread \
	$(CFLAGS) $^ $(CUDA_LIBPATH) $(CUDA_INC) $(INC) $(LDFLAGS_SHARED) -lcudart 
	$(BUILD) dpdkGPUNF
	@mv *.o $(OBJ_DIR)

dpdk_gpunf.o: $(SRC_DIR)/dpdk_gpu.c
	@gcc -g -w -m64 -pthread $(CFLAGS) $(LDFLAGS_SHARED) -o $@ -c $^ $(INC) 
	$(BUILD) dpdk_gpu

maingpunf.o: ./main/maingpunf.c
	@gcc -g -w -o $@ -c $^ $(INC) $(CUDA_LIBPATH) $(CUDA_INC) $(CFLAGS) $(LDFLAGS) $(LDFLAGS_SHARED) -lcudart -lcuda -lcudadevrt -ldl -pthread -lnuma
	$(BUILD) maingpunf

#nf compile
router.o: $(APP_DIR)/router/router.cu 
	@nvcc -g -w -arch=compute_$(SMV) -c $^ $(INC) $(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) router

nids.o: $(APP_DIR)/nids/nids.cu 
	@nvcc -g -w -arch=compute_$(SMV) -c $^ $(INC) $(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) nids

ipsec.o: $(APP_DIR)/ipsec_gw/ipsec.cu 
	@nvcc -g -w -arch=compute_$(SMV) -c $^ $(INC) $(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) ipsec

#gpu general compile
gpu_forwarder.o: $(SRC_DIR)/gpu_forwarder.cu 
	@nvcc -g -w -arch=compute_$(SMV) -c $^ $(INC) $(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) gpu_forwarder

maingpu.o: ./main/maingpu.c
	@gcc -g -w -o $@ -c $^ $(INC) $(CUDA_LIBPATH) $(CUDA_INC) $(CFLAGS) $(LDFLAGS) $(LDFLAGS_SHARED) -lcudart -lcuda -lcudadevrt -ldl -pthread -lnuma
	$(BUILD) maingpu

#general code compile
util.o : $(SRC_DIR)/util.c
	@gcc -w -m64 -pthread \
 	$(CFLAGS) $(LDFLAGS_SHARED) -o $@ -c $^ $(INC) $(CUDA_LIB_PATH) $(CUDA_INC)
	$(BUILD) util

clean:
	@rm -f $(OBJ_DIR)/*.o ./*.o dpdkGPU dpdkIO dpdkGPUNF dpdkGPUONE  
	@echo clean
else

ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable ")
endif

# Default target, detect a build directory, by looking for a path with a .config
RTE_TARGET ?= $(notdir $(abspath $(dir $(firstword $(wildcard $(RTE_SDK)/*/.config)))))

# LOG, CKJUNG
RTE_LOG_LEVEL=RTE_LOG_DEBUG
RTE_LIBRTE_ETHDEV_DEBUG=y

CFLAGS += -O3 $(shell $(PKGCONF) --cflags libdpdk)
LDFLAGS += $(shell $(PKGCONF) --libs libdpdk)

include $(RTE_SDK)/mk/rte.subdir.mk
include $(RTE_SDK)/mk/rte.extapp.mk
include $(RTE_SDK)/mk/rte.app.mk

endif
