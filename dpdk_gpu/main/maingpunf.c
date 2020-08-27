#include "dpdk_gpu.h"
#include "master.h"
#include "gpu_forwarder.h"
#include "util.h"

int main(int argc, char ** argv)
{
	set_gpu_mem_for_dpdk();

#if ROUTER
	initialize_router(1);
#elif NIDS
	initialize_nids(1);
#elif IPSEC
	initialize_ipsec(1);
#endif

	dpdk_handler(argc, argv);
	

	return 0;
}


