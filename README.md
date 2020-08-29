# GPU-Ether codes made by SH

* All source codes that made by SH are in this repository.
* These are used to evaluate GPU-Ether and DPDK.

---
## cachepollution

* NoisyNeighbor code is in this directory.
* Matrix Product is adopted for NosiyNeighbor.
* It proceeds the product of two matrixes to third matrix.
* The size of each matrix is 1140 X 1140.
	* This is same as the size of L3 cache size\(15MB\) in i7\-6800k CPU core.

---
## dpdk\_gpu

* DPDK codes for Packet I/O with GPU are in this directory.
* There are four versions of DPDK.
* These use 18.11.2 version of DPDKthem is reported in README of NF\_FOR\_GPU.
* All versions for DPDK can be excuted immediatly if users finished all settings about DPDK.
	* e.g.\) Load igb\_uio module.
* Detail description of these is reported in README of dpdk\_gpu.

---
## NF\_FOR\_GPU

* NF codes and pktgen configure files are in this directory.
* There are codes for DPDK and GPU\-Ether.
* Users can use these files only after serveral steps.
* Detail description of them is reported in README of NF\_FOR\_GPU.
