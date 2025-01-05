# PyGesture

## 목표

1. 손끝을 따라 움직이는 마우스 커서 이동 이벤트 만들기.
   - 검지와 중지가 모아져 있을 때만 커서 이동 이벤트 진행
   - opencv 창에서 현재 마우스 커서 위치를 표시할 수 있게 원형 도형 생성
2. 검지와 중지를 벌리면 마우스 클릭 이벤트 작동하게 하기.
   - 0.5초 전 검지 위치에서 클릭 이벤트 발동
   - 민감도 설정 가능하게 하기. 어느정도 벌려야 클릭이 작동하게 할지에 대해.


## 개선점

1. 검지와 중지가 모여있을 때의 기준이 애매함. 현재는 양 끝 말단이 카메라 정면 기준으로 겹쳐 있어야 마우스 이동 이벤트 작동.
   - 기준으로 생각해 볼 지점
     - 중지의 두번째 마디, 검지의 끝, 중지의 끝을 기점으로 하는 삼각형
     - 손바닥과 검지, 중지의 연결 부위의 거리
     - 손바닥의 기울기
     - 손 전체의 모양
       - 손가락 말단이 손바닥에 쥐어있는 상태로 구분
2. 클릭 이벤트 작동 트리거 변경
   - 실제로 해 보니까, 엄지와 약지의 말단이 붙을 경우 클릭 이벤트를 주는 편이 더 편했음. 벌릴 때 마우스 커서의 이동이 발생할 수 있기 때문에 트리거를 변경해야 함.

## 추가 목표

1. 양손을 다 쓸 수 있게끔 하기(완)
2. api를 제작해서 커스텀 제스처를 만들 수 있게 하기
3. 겹치는 원 범위(민감도)를 조절할 수 있게 하기
4. 이벤트 검출 시점과 이벤트 발동 시점을 분리하기

## 중간 목표
- 오른손으로 마우스 이동 및 클릭 가능하게 처리.(완)
- 왼손으로 q, w, e, r, d, f, b, 1, 2, 3, 4 입력 가능하게 처리.
- LOL 할 때 사용하는 키


### 추가 개선
- 점과 점이 겹칠 때 처리하는 방식의 문제
  - 생각보다 인식률이 좋지 않음.
  - 손의 각도에 따라서 오락가락 함
- 개선 방향: 지점 인식이 아닌, 구간 인식으로 전환
  - 현재는 특정 포인트가 인식 포인트에 가까이 가야 인식되게 했음
  - 개선 후에는 손 끝이 손바닥 내부로 들어가야 인식하게 할 것.
    - 손 끝의 위치로 구분하는 마우스 이동 이벤트는 그대로 진행.
    - 버튼 인식만 구간 인식으로 변경
  - 구간 인식의 장점
    - 현재는 각 좌표에 현재 스크린 크기값을 곱한 값의 거리를, distance라는 변수로 사용하고 있다.
    - 이 경우, 손이 가까워지고 멀어지는 것에 큰 영향을 받는다.
    - 손가락으로 할 수 있는 가장 확실한 표현은 굽히고 펴는 것이기 때문에, 정확도를 위해 굽히는 것의 정도는 구별하지 않고 0/1로 인식하려 한다.
    
    
## 설계 변경

여러 서비스를 이용할 수 있게, 서버 - 클라이언트 구조로 설계 변경

### 서버

- 회원 기능
- 제스처 저장

### 클라이언트 

- 제스처 작동
- 제스처 관리

## api 구현
- 이벤트 트리거 구성 요소(포인트, 트리거 등) 정보 제공
- 이벤트 트리거 저장/수정/삭제
- 로그인
