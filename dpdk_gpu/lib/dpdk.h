#ifndef __DPDK_H_
#define __DPDK_H_

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>

#include <rte_memory.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_mbuf.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_per_lcore.h>
#include <rte_launch.h>
#include <rte_ethdev.h>
#include <rte_ether.h>

#define NUM_MBUFS_DEFAULT 8192
#define MBUF_CACHE_SIZE 256
#define RX_DESC_DEFAULT 512
#define DEFAULT_PKT_BURST 32 // Increasing this number consumes memory very fast
#define LOOP_NUM 10

#define rte_ctrlmbuf_data(m) ((unsigned char *)((uint8_t*)(m)->buf_addr) + (m)->data_off)

#endif

