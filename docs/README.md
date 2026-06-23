# 지켜줘, 골든-타임

지능형 CCTV의 위험 객체 조기 탐지와 자율주행 드론의 초기 대응을 통합한 공공 안전 시스템의 프로토타입이다.

## 구성

- `cctv/`: CCTV 객체 탐지, 시계열 검증, 출동 권고
- `integration/`: 사건 메시지, 관제자 승인, 드론 임무 메시지
- `drone/`: 위험 인지형 경로 계획, 이상 감지, 비상 대응
- `docs/`: 설계 문서와 참고 자료

## 주요 노트북

1. `integration/notebooks/Public_Safety_EndToEnd_Demo.ipynb`
2. `cctv/notebooks/CCTV_Public_Safety_Detection_Colab.ipynb`
3. `drone/notebooks/RiskAware_Path_Stages_Colab.ipynb`
4. `drone/notebooks/FullIntegratedDroneSafety_Colab.ipynb`
5. `drone/notebooks/EmergencySequencer_Colab.ipynb`


