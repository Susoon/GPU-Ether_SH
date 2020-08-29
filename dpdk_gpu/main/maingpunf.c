#include "dpdk_gpu.h"
#include "master.h"
#include "gpu_forwarder.h"
#include "util.h"

int main(int argc, char ** argv)
{
	set_gpu_mem_for_dpdk();

#if ROUTER
	initialize_router();
#elif NIDS
	initialize_nids();
#elif IPSEC
	initialize_ipsec();
#endif

	dpdk_handler(argc, argv);
	

	return 0;
}


