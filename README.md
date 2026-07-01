## 5. 전체 시스템 아키텍처

```text
[Camera Frame]
   ↓ MediaPipe FaceMesh / FaceLandmarker (478 landmarks + pose)
   ↓ Segmentation (ONNXRuntime CUDA)
   ↓ Post-process (stabilization + color transform + mask caching)
   ↓ 2D Makeup Render (skin / smoky eye / iris / sclera / lip)
   ↓ Face Pattern Render (OpenGL UV texture mapping)
   ↓ Composite (mask / occlusion / background)
   ↓ Display (OpenGL + Pygame window)
```

### CPU/GPU 역할 분리 전략

| 담당 | 역할 |
| --- | --- |
| GPU | 세그멘테이션 추론, OpenGL 텍스처 렌더링, 삼각형 보간/래스터라이즈, 대규모 픽셀 연산 |
| CPU | 랜드마크 안정화, 얼굴/마스크 영역 계산, 색공간 변환, 마스크 후처리, 물리 상태 업데이트 |

핵심 전략은 **무거운 추론/픽셀/보간 작업은 GPU에 최대한 고정하고, 좌표·상태·후처리 연산은 CPU에서 가볍게 처리**하는 것입니다.

---

## 6. 성능 최적화 및 트러블슈팅

### 6.1 CPU 기반 2D 워핑 병목

#### 문제

초기 구조에서는 얼굴 패턴을 삼각형 단위로 나누고 각 삼각형마다 `cv2.warpAffine`을 호출했습니다. 478개 랜드마크 기반 삼각형을 CPU에서 반복 보간/래스터라이즈하면서 프레임 드랍이 발생했습니다.

#### 해결

- 화면 좌표(VBO)와 UV 좌표를 GPU로 전달
- OpenGL `glDrawArrays(GL_TRIANGLES)` 기반 텍스처 매핑 구조로 전환
- 텍스처 업로드 시 `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)`로 줄틀림 문제 해결
- OpenCV(BGR)와 OpenGL(RGB) 색공간 차이 보정
- `GL_CLAMP_TO_EDGE` 설정으로 경계 찢김 완화

#### 결과

- 전체 프레임 시간: **156ms → 78.4ms**
- FPS: **6~7 FPS -> 12~13 FPS**
- warp 처리 비용: **약 35% 감소**

---

### 6.2 세그멘테이션 추론 비용 과다

#### 문제

ONNX 기반 세그멘테이션을 매 프레임 전체 영상에 수행하면 피부 보정과 배경 합성 단계에서 큰 병목이 발생했습니다.

#### 해결

- CUDAExecutionProvider 우선 활성화
- N프레임당 1회 추론 후 마스크 캐싱
- 얼굴 중심/connected component 기반 대상 인물 필터링
- soft mask와 morphology 후처리로 품질 유지
- I/O Binding 적용 가능 환경에서는 CPU-GPU 복사 최소화

#### 결과

- 피부 세그멘테이션 비용: **약 80% 이상 감소**
- 프레임 처리 변동폭 감소로 실시간성이 개선됨

---

### 6.3 랜드마크 떨림

#### 문제

실시간 카메라 입력에서는 얼굴 랜드마크가 프레임마다 미세하게 흔들려 메이크업/패턴이 떨려 보이는 문제가 있었습니다.

#### 해결

- EMA(Exponential Moving Average) 기반 smoothing
- 프레임 간 급격한 이동 제한(clamping)
- 눈 뜸/감김 상태를 threshold와 smoothing으로 추적해 눈 효과 오작동 완화

---

### 6.4 색상 왜곡 및 마스크 경계 깨짐

#### 문제

OpenCV는 BGR, OpenGL은 RGB 중심으로 처리되어 색상이 뒤바뀌거나, 세그멘테이션 경계가 거칠게 보이는 문제가 있었습니다.

#### 해결

- 렌더링 전 색공간 변환 적용
- GaussianBlur 기반 feathering
- morphological closing/opening으로 작은 구멍/잡티 정리
- temporal EMA로 배경/사람 마스크 깜빡임 완화

---

## 7. 왜 Python으로 구현했는가

- 교육 과정에서 Python을 메인 언어로 사용해 개발 속도와 실험 효율이 높았습니다.
- OpenCV, MediaPipe, ONNXRuntime, PyOpenGL 등 실험에 필요한 라이브러리를 빠르게 연결할 수 있었습니다.
- TensorRT 엔진화는 레이어 호환성, 플러그인 개발, 엔진 빌드 시간이 추가로 필요했습니다.
- 프로젝트 일정과 목표 성능을 고려했을 때 ONNXRuntime CUDAExecutionProvider만으로도 실시간 목표에 도달할 수 있어 ROI 관점에서 Python + ONNXRuntime 구조를 유지했습니다.

---

## 8. 실행 방법

### 8.1 의존성 설치

```bash
pip install -r requirements.txt
```

> Windows GPU 환경에서는 CUDA/cuDNN 및 `onnxruntime-gpu` provider가 정상 인식되는지 확인하세요.

### 8.2 실행

```bash
python main.py
```

### 8.3 기본 조작

- `ESC`: 프로그램 종료

---

## 9. 프로젝트 구조

```text
.
├─ main.py                 # 메인 루프, 카메라 입력, 효과 적용, 렌더링 orchestration
├─ config.py               # 메이크업/세그멘테이션/렌더링/3D/물리 파라미터
├─ face_effects.py         # 2D 메이크업 효과와 ONNX 세그멘테이션 처리
├─ face_processing.py      # 얼굴 검출, 랜드마크, jaw/occlusion mask 생성
├─ pose_processing.py      # 포즈 기반 owner ROI mask 생성
├─ rendering.py            # OpenGL 텍스처/마스크/3D 렌더링 유틸
├─ shaders.py              # OpenGL shader program 정의
├─ sticker3d/renderer.py   # 얼굴 패턴 UV 텍스처 렌더링
├─ physics.py              # 보조 3D 스트랩용 Verlet rope physics
├─ asset/                  # ONNX 모델, UV 좌표, triangle index, 배경/텍스처 리소스
├─ patterns/               # 얼굴 패턴 PNG 리소스
├─ obj/, models/           # MediaPipe/3D 모델 리소스
└─ requirements.txt        # Python 의존성
```

---

## 10. 포트폴리오에서 강조할 점

- **실시간 2D 메이크업 파이프라인 설계**: 피부/눈/입술/패턴이 얼굴 랜드마크를 따라 자연스럽게 움직이도록 구현
- **성능 병목을 수치 기반으로 개선**: 156ms → 78ms, 6~7 FPS → 12~13 FPS로 개선
- **CPU/GPU 역할 분리 경험**: CPU 워핑 병목을 OpenGL 텍스처 매핑으로 대체
- **AI 추론 최적화 경험**: ONNXRuntime CUDA provider, 추론 주기 제어, 마스크 캐싱, 대상 인물 게이팅 적용
- **실시간 시스템 관점의 문제 해결**: 색공간 차이, 마스크 경계, 랜드마크 떨림, 다중 인물 환경을 고려한 안정화

---

## 11. 한계 및 개선 방향

- Unity/Unreal 같은 실시간 엔진으로 이전하면 배포와 UI/효과 제작 파이프라인을 더 쉽게 구성할 수 있습니다.
- C++ 또는 CUDA 커스텀 커널로 주요 연산을 이전하면 추가 성능 개선이 가능합니다.
- TensorRT를 적용하면 추론 속도 개선 여지가 있지만, 모델 호환성 검증과 엔진 빌드 자동화가 필요합니다.
- 효과 preset UI, 파라미터 실시간 조정 도구, 데모 녹화 기능을 추가하면 포트폴리오 전달력을 높일 수 있습니다.

---

## 12. 느낀 점

본 프로젝트를 통해 실시간 시스템에서는 기능 구현만큼 **구조 설계와 성능 계측**이 중요하다는 점을 체감했습니다.

특히 CPU 기반 워핑 병목을 로그와 프레임 시간으로 분석하고, GPU 렌더링 구조로 전환하는 과정에서 문제를 감이 아닌 수치로 정의하고 개선하는 접근 방식을 익혔습니다. 또한 MediaPipe, ONNXRuntime, OpenGL, OpenCV를 하나의 파이프라인으로 연결하며 실시간 AR 필터가 어떤 지점에서 병목을 만들고 어떻게 안정화해야 하는지 경험할 수 있었습니다.
