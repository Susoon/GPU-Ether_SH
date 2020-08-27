# NF Evaluation for GPU

* GPU를 위한 NF코드
* ipsec\_gw : IPSec Gateway 코드
	* 수정 필요
* nids : NIDS 코드
* router : Router 코드
* lib : NF들이 사용하는 헤더파일과 input 파일들
* DPDK : DPDK와 함께 사용된 NF 파일들
	* 위의 폴더들이 내장되어 있다.
* GPU\-Ether : GPU-Ether와 함께 사용된 NF 파일들
	* 위의 폴더들이 내장되어 있다.
* RandPktGen : 각 NF를 위한 pktgen의 랜덤패킷 전송을 위한 configure 파일과 sh 파일

---
## 사용법

* RandPktGen 
	* 내부의 파일과 폴더들을 모두 pktgen폴더에 옮긴뒤 sh파일을 실행
* NF 코드
	* tx\_kernel의 extract\_buf함수의 app\_idx 인수를 1이아닌 2로 변경
	* main함수에서 initialize\_\(NF명\)\(\(Mempool변수명\), \(pkt count 배열 변수명\),1\)을 호출
		* e.g.) initialize\_router(d\_mempool, pkt\_cnt, 1);

