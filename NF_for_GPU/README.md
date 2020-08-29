# NF Evaluation codes for GPU

---
## Description

* This repository has code files that are used to evaluate NF performance in GPU.
* There are three NF codes and four configure files in this repo.
* Thread numbers of kernels that are used for each NF can be differ because of packet size.

---
### Router

* DIR\-24\-8\-BASIC algorithm is adopted.
* For all packet sizes, same number of threads are used.
	* One thread per One packet.
* The performance is the same as the forwarding rate for DPDK and GPU\-Ether.

---
### NIDS

* NIDS is implemented based on the codes from snort.
* For each packet sizes, different number of threads used.
	* The number of threads used in each packet sizes is recoreded in pkt\_data.h.
* The performance is the same as the forwarding rate for DPDK and GPU\-Ether.

---
### IPSec gateway

* AES 128bit ver. and SHA1 are adopted.
* For each packet sizes, different number of threads used.
	* The number of threads used in each packet sizes is recoreded in pkt\_data.h.
* The performance is the same as the forwarding rate for DPDK.
	* This is because the forwarding rate of DPDK is lower than the maximum performance of IPSec gateway.
* IPSec coupled to GPU-Ether showed lower performance than the forwarding rate.

---
### RandPktGen

* There are four pktgen configure files in this directory.
* Each configure file is for NFs and the forwarding evaluation.
* default.pkt
	* Configure for the forwarding.
	* There are configures about setting source and destination ip and destination mac address.
* router.pkt
	* Configure for the router.
	* It enable range mode of pktgen.
	* This file allows users to randomly set the source and destination ip and port of the packet.
		* If the port is set to inc 0, it is set randomly.
	* In range mode, users have to set packet size in 2 ways when run pktgen.
		* For range mode and for set mode.
		* e.g.\)
			* range 0 size start 64
			* range 0 size min 64
			* range 0 size max 64
			* set 0 size 64
* nids.pkt
	* Configure for the nids.
	* It enable range mode of pktgen.
	* This file allows users to randomly set the destination port and payload of the packet.
	* The way how to set packet size is same with router.pkt case.
* ipsec.pkt
	* Confiure for the IPSec gateway.
	* It enable range mode of pktgen.
	* This file allows users to randomly set all component of the packet except the mac address.
	* The way how to set packet size is same with router.pkt case.
	
---
## How to use

* NF codes

1. DPDK
	* Call initialize initialize\_\(NF name\)\(\) in main function.
		* e.g.\) initialize\_router\(\)

2. GPU\-Ether
	* Call initialize initialize\_\(NF name\)\(\(variable name of Mempool\), \(variable name of pkt count array\)\) in main function.
		* e.g.\) initialize\_router\(d\_mempool, pkt\_cnt\)
	* Change app\_idx parameter value of extract\_buf function in tx\_kernel 1 to 2.
		* tx\_kernel function is in lib/core.cu.

* RandPktGen
	* Copy all files in the directory to pktgen directory and excute corresponding sh file.

