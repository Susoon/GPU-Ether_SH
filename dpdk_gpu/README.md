#DPDK Evaluation Source Code

* 5 version of DPDK for Evaluation with GPU

---
## Component

1. src
	* All source codes are included which is needed for each DPDK version.
	* Files started with dpdk\_ are the code of DPDK functionalities.
	* The util.c file is a file that contains functions and macros commonly used in all dpdk versions.

2. lib 
	* Header files that is used by DPDK source codes are included.
	* Composition of codes are same with src folder.

3. main
	* Each DPDK version need different main function.
	* All main functions are included.

4. apps
	* NF source codes are included.
	* All codes works successfully.
	* By changing the value in pkt\_data.h, you can control the number of threads that used by each NF.

---
## tunnelQ

* Queue that is used for communication between Master thread and Worker thread.
* Worker\-Master version DPDK use tunnelQ.
* It is a structure that has RXQueue, TXQueue, rhead and head.
	* rhead is a head of RXQueue and thead is a head of TXQueue.

---
## Description of DPDK

---
### 1. dpdkIO

* dpdkIO can be compiled and excuted by dpdkio.sh.
* It use **util.c dpdkio.c mainio.c**.
	* To use above files, it use **util.h dpdkio.h**.
* All I/O processes proceed only in CPU.
* Macro GPU that is defined in util.c must be set 0.

---
### 2. dpdkGPU

* dpdkGPU can be compiled and excuted by dpdkgpu.sh
* It use **master.c gpu_forwarder.c dpdk_gpu.c util.c maingpu.c**.
	* To use above files, it use **master.h gpu_forwarder.h dpdk_gpu.h util.h**.
* Packet transmission proceeds to NIC \-> CPU\(DPDK\_) \-> GPU.
* Packet I/O\(DPDK\) takes Worker\-Master scheme.
* Macro GPU that is defined in util.c must be set 1.

---
### 3. dpdkGPUONE

* dpdkGPUONE can be compiled and excuted by dpdkgpuone.sh
* It use **gpu_forwarder.c dpdk_gpuone.c util.c maingpu.c**.
	* To use above files, it use **gpu_forwarder.h dpdk_gpuone.h util.h**.
* Packet transmission proceeds to NIC \-> CPU\(DPDK\) \-> GPU.
* All Packet I/O processes proceed in one thread.
* Macro GPU that is defined in util.c must be set 1.

---
### 4. dpdkGPUNF

* dpdkGPUNF can be compiled and excuted by dpdkgpunf.sh
* It use **gpu_forwarder.c dpdk_gpu.c util.c maingpunf.c router.cu nids.cu ipsec.cu**.
	* To use above files, it use **gpu_forwarder.h dpdk_gpu.h util.h router.h nids.h ipsec.h gf\_tables.h  sbox.h pkt_data.h**
* Packet transmission proceeds to NIC \-> CPU\(DPDK\_) \-> GPU.
* Packet I/O\(DPDK\) takes Worker\-Master scheme.
* To excute each NF, initializer function must be uncommented.
	* All intializer functions are in maingpunf.c
	* Each initializer function has the following form.
		* initialize\_router\(1\); 
* Macro GPU that is defined in util.c must be set 1.

