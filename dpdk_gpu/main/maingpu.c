#include "dpdk_gpu.h"
#include "master.h"
#include "gpu_forwarder.h"
#include "util.h"

int main(int argc, char ** argv)
{
	set_gpu_mem_for_dpdk();

	gpu_monitor_loop();

	dpdk_handler(argc, argv);
	

	return 0;
}
