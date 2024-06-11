# Chord-Vision

## Link 
- GITHUB: https://github.com/zeberity123/chord-vision.git

## 프로젝트 개요
- 유튜브에서 기타를 배우려는 사람들은 종종 자신이 좋아하는 노래의 기타 튜토리얼을 찾지만, 때로는 튜토리얼 영상보다는 유명 기타리스트의 커버 영상을 발견할 때가 있습니다. 그러나 대부분의 커버 영상에는 코드나 채보가 영상에 나와 있지 않습니다. 이러한 상황에서, 해당 기타리스트의 손 모양을 영상만으로 분석하고 코드를 식별하여 직접 따라서 배우고자 하는 사람들을 위해 이 프로젝트가 시작하였습니다.
  
## 프로젝트 목표
- 유튜브 영상을 크롤링하여 기타 연주 영상에서 손 모양이 잡고 있는 기타 코드를 식별하고, 이를 사용자에게 알려주는 모델을 개발하는 것을 목표로 합니다.

- 웹캠 및 노트북 카메라 등을 활용하여 실시간으로 코드를 식별하고 이를 사용자에게 알려줍니다.



## Requirements

## Installation
Download the latest version from [here](https://github.com/zeberity123/chord-vision/releases).

```
cd chord-vision-1.0.0
pip install -r requirements.txt
```

## How to run
- 동영상의 경우:
```
python3 video_run.py
```

- 라이브 카메라의 경우:
```
python live_run.py
```

## Resources
하드웨어
- 한컴아카데미 내 PC
- 한컴아카데미 지원 노트북
- Nvidia AGX Xavier
- Logitech Webcam

소프트웨어 및 프레임워크
- Github: https://github.com/zeberity123/chord-vision
- MediaPipe
- opencv_python
- youtube_dl
- torchvision
- Pytorch
- Numpy
- Tensorflow

  

## Team members
- 김현우 darudayu123@gmail.com
- 백동렬 dbstickman@gmail.com
- 장지완 jiwanjang0577@gmail.com
