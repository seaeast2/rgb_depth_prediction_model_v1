# rgb_depth_prediction_model_v1

RGB 이미지에서 상대 깊이(relative depth)를 추정하는 간단한 baseline 모델입니다.
단일 이미지에서는 카메라 정보 없이 실제 미터 단위 깊이를 복원할 수 없으므로,
출력값은 가까운 픽셀일수록 1에 가까운 0~1 범위의 상대값입니다.

## 설치

```bash
python -m pip install -e .
```

## Python 사용법

```python
from rgb_depth_prediction import RGBDepthEstimator

estimator = RGBDepthEstimator()
depth = estimator.predict("input.jpg")       # float32 배열, (height, width)
estimator.predict_image("input.jpg").save("depth.png")
```

## CLI

```bash
rgb-depth input.jpg depth.png
```

이 구현은 외부 가중치가 필요 없는 실행 가능한 baseline이며, 향후 DUST3R,
MASt3R, VGGT 계열의 학습 모델로 `RGBDepthEstimator.predict` 내부를 교체할 수
있도록 동일한 RGB 입력/깊이 출력 인터페이스를 제공합니다.
