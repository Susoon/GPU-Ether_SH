#ifndef __FORWARDER_H_
#define __FORWARDER_H_

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include "dpdk.h"
#include "util.h"

void forwarder(void *data);

#endif
