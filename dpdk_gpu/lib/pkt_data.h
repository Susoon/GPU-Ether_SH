#ifndef __PKT_DATA_H_
#define __PKT_DATA_H_

#define CHAPTER_NUM 4

#define RX_NB 32

#define ONELINE 6

#define IPSEC 0 

#define PKT_DATA_SIZE 60
#define HEADROOM_SIZE 64
#define TAILROOM_SIZE HEADROOM_SIZE + 4
#define PKT_SIZE (HEADROOM_SIZE + PKT_DATA_SIZE + TAILROOM_SIZE)

#if IPSEC
#define IPSECHEAD 28 
#define IPSECTAIL 22
#else
#define IPSECHEAD 0
#define IPSECTAIL 0
#endif

#define ROUTER_T_NUM 512
#define ROUTER_TB_NUM 1

#if PKT_DATA_SIZE == 60

#if IPSEC
#define NF_T_NUM 768
#define NF_TB_NUM 2
#define THD_PER_PKT 3
#define DATA_PER_THD 16

#else
#define NF_T_NUM 512
#define NF_TB_NUM 1
#define THD_PER_PKT 1
#define DATA_PER_THD 18
#endif

#elif PKT_DATA_SIZE == 124
#define NF_T_NUM 1024
#define NF_TB_NUM 2
#define THD_PER_PKT 4
#define DATA_PER_THD 32

#elif PKT_DATA_SIZE == 252
#define NF_T_NUM 854
#define NF_TB_NUM 3
#define THD_PER_PKT 5
#define DATA_PER_THD 48

#elif PKT_DATA_SIZE == 508
#define NF_T_NUM 1024
#define NF_TB_NUM 4
#define THD_PER_PKT 8
#define DATA_PER_THD 64

#elif PKT_DATA_SIZE == 1020
#define NF_T_NUM 951
#define NF_TB_NUM 7
#define THD_PER_PKT 13
#define DATA_PER_THD 80

#elif PKT_DATA_SIZE == 1460
#define NF_T_NUM 1024
#define NF_TB_NUM 8
#define THD_PER_PKT 16
#define DATA_PER_THD 96

#elif PKT_DATA_SIZE == 1510
#define NF_T_NUM 1024
#define NF_TB_NUM 8
#define THD_PER_PKT 16
#define DATA_PER_THD 96
#endif

#define PAD_LEN 0

#define PKT_BATCH_NUM (512 + RX_NB) 
#define PKT_BATCH_SIZE (PKT_SIZE * PKT_BATCH_NUM)

#define POLL 1

#endif
