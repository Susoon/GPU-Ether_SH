#ifndef __UTIL_H_
#define __UTIL_H_

#include <stdio.h>
#include <unistd.h>
#include <memory.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include "dpdk.h"
#include "pkt_data.h"

#define ONE_SEC (uint64_t)(1000*1000*1000)
#define PKT_MONITORING 1

#define START_RED printf("\033[1;31m");
#define START_GRN printf("\033[1;32m");
#define START_YLW printf("\033[1;33m");
#define START_BLU printf("\033[1;34m");
#define END printf("\033[0m"); 

#define MEGA 1000 * 1000

struct tunnelQ{
	int rhead;
	int thead;
	unsigned char* pktRXQ;
	unsigned char* pktTXQ;
};

struct rte_mbuf;

uint64_t monotonic_time(void);
void swp_hdr(unsigned char* ptr);
void make_char_buf(struct rte_mbuf *buf[], unsigned char *pktQ, int head, int nb);
void make_rte_buf(unsigned char *pktQ, struct rte_mbuf **buf, int head, int nb);
void swp_hdr_buf(struct rte_mbuf **buf, int head, int nb);
void *cpu_monitoring_loop(void *data);
void print_cur_stat(const char ** u_str);
void print_pkt(unsigned char *ptr, int idx);

#endif
