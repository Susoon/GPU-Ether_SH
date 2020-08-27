#include "gpu_forwarder.h"

unsigned char *pktBuf;
int *nbBoard;
int *statusBoard;
int * pkt_cnt;

#define TX 1

extern void print_pkt(unsigned char * ptr, int idx);

__global__ void gpu_monitor_launch(unsigned char *pktBuf, int *nbBoard, int *statusBoard, int * pkt_cnt);

__device__ void print_gpu(unsigned char* buf, int idx)
{
	int i;
	START_YLW
	printf("[GPU]: %dth pkt dump: \n", idx);
	for(i = 0; i < PKT_SIZE; i++)
	{
		if(i != 0 && i % ONELINE == 0)
			printf("\n");
		printf("%02x ", buf[i]);
	}
	printf("\n");
	END
}

extern "C"
void copy_to_gpu(unsigned char* buf, int nb)
{
	static int idx = 0;

	int status[CHAPTER_NUM];
	int nba[CHAPTER_NUM];
	ASSERTRT(cudaMemcpy(status, statusBoard, sizeof(int)*CHAPTER_NUM, cudaMemcpyDeviceToHost));

	if(status[idx] != 0)
		return;

	if(nb > 512)
		nb = 512;
	ASSERTRT(cudaMemcpy(pktBuf + (idx * PKT_BATCH_SIZE), buf, sizeof(unsigned char) * nb * PKT_SIZE, cudaMemcpyHostToDevice));
	ASSERTRT(cudaMemcpy(nbBoard + idx, &nb, sizeof(int), cudaMemcpyHostToDevice));
	ASSERTRT(cudaMemset(statusBoard + idx, -1, sizeof(int)));

	idx++;
	idx %= CHAPTER_NUM;	
}

extern "C"
int copy_from_gpu(unsigned char* buf){

	static int idx = 0;
	int tmp = 0;
	int status[CHAPTER_NUM];
	int nb[CHAPTER_NUM];

	ASSERTRT(cudaMemcpy(status, statusBoard, sizeof(int)*CHAPTER_NUM, cudaMemcpyDeviceToHost));
	ASSERTRT(cudaMemcpy(nb, nbBoard, sizeof(int)*CHAPTER_NUM, cudaMemcpyDeviceToHost));

	if(status[idx] != nb[idx] || nb[idx] == 0)
		return 0;

  	ASSERTRT(cudaMemset(nbBoard + idx, 0, sizeof(int)));
	ASSERTRT(cudaMemcpy(buf, pktBuf + (idx * PKT_BATCH_SIZE), sizeof(unsigned char)*nb[idx]*PKT_SIZE, cudaMemcpyDeviceToHost));
  	ASSERTRT(cudaMemset(statusBoard + idx, 0, sizeof(int)));

	tmp = idx;	

	idx++;
	idx %= CHAPTER_NUM;

	return nb[tmp];
}

extern "C"
void set_gpu_mem_for_dpdk(void)
{
	START_BLU
	printf("_______________DPDK________________\n");
	printf("RING_SIZE = %lld\n", RING_SIZE);
	printf("PKT_SIZE = %d, PKT_BATCH_NUM = %d\n", PKT_SIZE, PKT_BATCH_NUM);
	END

	ASSERTRT(cudaMalloc((void**)&pktBuf, RING_SIZE));
  	ASSERTRT(cudaMemset(pktBuf, 0, RING_SIZE));
	printf("GPU Descriptor pktBuf Setting Done!\n");

	ASSERTRT(cudaMalloc((void**)&statusBoard, sizeof(int)*CHAPTER_NUM));
  	ASSERTRT(cudaMemset(statusBoard, 0, sizeof(int)*CHAPTER_NUM));
	printf("GPU Descriptor StatusBoard Setting Done!\n");

	ASSERTRT(cudaMalloc((void**)&nbBoard, sizeof(int)*CHAPTER_NUM));
  	ASSERTRT(cudaMemset(nbBoard, 0, sizeof(int)*CHAPTER_NUM));
	printf("GPU Descriptor nbBoard Setting Done!\n");

	ASSERTRT(cudaMalloc((void**)&pkt_cnt, sizeof(int)));
  	ASSERTRT(cudaMemset(pkt_cnt, 0, sizeof(int)));
	printf("pkt_cnt Setting Done!\n");

	START_GRN
	printf("[Done]____GPU mem set for dpdk____\n");
	END
}

extern "C"
int get_gpu_cnt(void)
{
	int cur_pkt = 0;

	ASSERTRT(cudaMemcpy(&cur_pkt, pkt_cnt, sizeof(int), cudaMemcpyDeviceToHost));
	ASSERTRT(cudaMemset(pkt_cnt, 0, sizeof(int)));	

	return cur_pkt;
}

__global__ void gpu_monitor_poll(unsigned char *pktBuf, int *nbBoard, int *statusBoard, int * pkt_cnt)
{
	__shared__ uint8_t chapter_idx;
	__shared__ int nb;

	if(threadIdx.x == 0){
		chapter_idx = 0;
		nb = 0;
	}

	while(true){
		__syncthreads();
		if(threadIdx.x == 0 && statusBoard[chapter_idx] == -1){
			nb = nbBoard[chapter_idx];
		}

		__syncthreads();		
		if(threadIdx.x < nb){
			atomicAdd(&statusBoard[chapter_idx], 1);
			atomicAdd(pkt_cnt, 1);
			if(threadIdx.x == 0){
#if TX
				atomicAdd(&statusBoard[chapter_idx], 1);
#else
				statusBoard[chapter_idx] = 0;
				nbBoard[chapter_idx] = 0;
#endif
				nb = 0;
			}
		}

		__syncthreads();
		if(threadIdx.x == 0){
			chapter_idx++;
			chapter_idx %= CHAPTER_NUM;
		}
	}
}

extern "C"
void gpu_monitor_loop(void)
{
	cudaStream_t stream;
	ASSERT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	gpu_monitor_poll<<<1, THREAD_NUM, 0, stream>>>(pktBuf, nbBoard, statusBoard, pkt_cnt);
}

