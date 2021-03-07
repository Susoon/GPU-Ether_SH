#include "dpdk_gpu.h"
#include "master.h"
#include "gpu_forwarder.h"
#include "util.h"

int main(int argc, char ** argv)
{
	set_gpu_mem_for_dpdk();

	initialize_router(1);
//	initialize_nids(1);
//	initialize_ipsec(1);

	dpdk_handler(argc, argv);
	

	return 0;
}


