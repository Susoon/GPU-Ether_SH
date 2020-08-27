#ifndef __DPDK_GPU_H_
#define __DPDK_GPU_H_

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>

#include <rte_memory.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_per_lcore.h>
#include <rte_mbuf.h>
#include <rte_launch.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include "l2p.h"
#include "util.h"

l2p_t *l2p;

static struct rte_eth_conf default_port_conf = {
rxmode : {       
		split_hdr_size : 0,
	},                        
txmode : {                                    
mq_mode : ETH_MQ_TX_NONE,      
	}, 
};  

int launch_one_lcore(void *arg __rte_unused);
static __inline__ void start_lcore(l2p_t *l2p, uint16_t lid)
{
	l2p->stop[lid] = 0;
}


static __inline__ int32_t lcore_is_running(l2p_t *l2p, uint16_t lid)
{
	return l2p->stop[lid] == 0;
}

static void IOloop(uint8_t lid);

typedef struct rte_eth_stats eth_stats_t;

typedef struct port_info{
	eth_stats_t prev_stats;
	eth_stats_t curr_stats;
}port_info_t;

typedef struct ck_dpdk{
	port_info_t info[RTE_MAX_LCORE];
}ck_dpdk_t;

ck_dpdk_t ck_dpdk; 

void rte_timer_setup(void);
static void * _timer_thread(void*);

void dpdk_handler(int argc, char **argv);
static void * _timer_thread(void *nothing);
void rte_timer_setup(void);

#endif

