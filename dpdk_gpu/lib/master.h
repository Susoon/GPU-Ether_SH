#ifndef __MASTER_H_
#define __MASTER_H_

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include "util.h"
#include "gpu_forwarder.h"

void master(void *data);

#endif
