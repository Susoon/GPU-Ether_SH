#include "gpu_forwarder.h"

//GPU가 사용하는 패킷 버퍼
unsigned char *pktBuf;
//GPU에 전송된 패킷 수
int *nbBoard;
//GPU가 가진 4배수 패킷 버퍼의 각각 챕터의 status
int *statusBoard;
//GPU가 카운트한 패킷 수
int * pkt_cnt;

#define DUMP 0
#define TX 1

extern void print_pkt(unsigned char * ptr, int idx);

__global__ void gpu_monitor_launch(unsigned char *pktBuf, int *nbBoard, int *statusBoard, int * pkt_cnt);

//하나의 패킷을 dump하는 device함수
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

//CPU가 GPU에 패킷을 전송하기위해 사용하는 함수
//주석처리된 부분은 statusBoard를 통해 가용한 패킷 버퍼를 확인하는 부분이다.
//주석처리를 해제할 경우 마지막에 idx를 변화시키는 부분을 제거해야한다.
//LAUNCH 매크로 안의 부분은 이전에 persistent kernel과 kernel launch를 비교하기위해
//구현한 부분이니 현재의 상황과 무관하다.
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
#if DUMP
	ASSERTRT(cudaMemcpy(nba, nbBoard, sizeof(int)*CHAPTER_NUM, cudaMemcpyDeviceToHost));
	printf("[%s] HELLO!! idx : %d, nb : %d %d %d %d, status : %d %d %d %d\n", __FUNCTION__, idx, nba[0], nba[1], nba[2], nba[3], status[0], status[1], status[2], status[3]);
#endif

#if LAUNCH
	cudaStream_t stream;
	ASSERTRT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	gpu_monitor_launch<<<1, THREAD_NUM, 0, stream>>>(pktBuf, nbBoard, statusBoard, pkt_cnt);
	cudaDeviceSynchronize();
	cudaStreamDestroy(stream);
#endif

	idx++;
	idx %= CHAPTER_NUM;	
}

//GPU로부터 패킷을 전송받는 함수이다.
//statusBoard를 통해 처리가 끝난 패킷을 확인후 복사해온다.
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

#if DUMP
	printf("[%s] HELLO!! idx : %d, nb : %d %d %d %d, status : %d %d %d %d\n", __FUNCTION__, idx, nb[0], nb[1], nb[2], nb[3], status[0], status[1], status[2], status[3]);
#endif
  	ASSERTRT(cudaMemset(nbBoard + idx, 0, sizeof(int)));
	ASSERTRT(cudaMemcpy(buf, pktBuf + (idx * PKT_BATCH_SIZE), sizeof(unsigned char)*nb[idx]*PKT_SIZE, cudaMemcpyDeviceToHost));
  	ASSERTRT(cudaMemset(statusBoard + idx, 0, sizeof(int)));

	tmp = idx;	

	idx++;
	idx %= CHAPTER_NUM;

	return nb[tmp];
}

//DPDK를 위한 메모리를 세팅하는 함수
extern "C"
void set_gpu_mem_for_dpdk(void)
{
	START_BLU
#if POLL
	printf("__________POLLING VERSION___________\n");
#else
	printf("__________KERNEL LAUNCH VERSION___________\n");
#endif
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

//GPU가 카운트한 패킷 수를 넘겨주는 함수
extern "C"
int get_gpu_cnt(void)
{
	int cur_pkt = 0;

	ASSERTRT(cudaMemcpy(&cur_pkt, pkt_cnt, sizeof(int), cudaMemcpyDeviceToHost));
	ASSERTRT(cudaMemset(pkt_cnt, 0, sizeof(int)));	

	return cur_pkt;
}

//launch형식의 GPU함수
//현재상황과 무관하다.
__global__ void gpu_monitor_launch(unsigned char *pktBuf, int *nbBoard, int *statusBoard, int * pkt_cnt)
{
	static uint8_t chapter_idx = 0;
	int nb = 0;

	if(threadIdx.x == 0 && statusBoard[chapter_idx] == 1){
		nb = nbBoard[chapter_idx];
		nbBoard[chapter_idx] = 0;
	}
	
	if(threadIdx.x < nb){
		atomicAdd(pkt_cnt, nbBoard[chapter_idx]);
	}
	if(threadIdx.x == 0){
		chapter_idx++;
		chapter_idx %= CHAPTER_NUM;
		statusBoard[chapter_idx] = 2;
	}
}

//현재 GPU에서 실행되는 함수
//statusBoard를 polling하며 RX한 패킷이 담기는지 확인
//받은 패킷 수(nb)보다 작은 번호를 가진 thread만 패킷을 확인
//4개의 chapter를 가짐
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
//			memset(&pktBuf[chapter_idx*PKT_BATCH_SIZE + threadIdx.x*PKT_SIZE + HEADROOM_SIZE], 0, 60);
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

//main함수에서 호출되는 함수
//GPU함수를 호출한다.
extern "C"
void gpu_monitor_loop(void)
{
	cudaStream_t stream;
	ASSERT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	gpu_monitor_poll<<<1, THREAD_NUM, 0, stream>>>(pktBuf, nbBoard, statusBoard, pkt_cnt);
}

