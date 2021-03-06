---
title: "[Tensorflow] Tensorflow 2.0 GPU 설치 (Window)"
date: 2020-05-13
categories:
- ETC
tags:
- Tensorflow
---

이 글에서는 **Tensorflow 2.2.0** 기준 설치 과정을 설명합니다.

### 환경 셋팅
1. Visual Studio 2017 설치 ([링크](https://docs.microsoft.com/ko-kr/visualstudio/releasenotes/vs2017-relnotes))
2. NVIDIA driver 설치 ([링크](https://www.nvidia.co.kr/Download/index.aspx?lang=kr))
3. Anaconda 3 설치 ([링크](https://www.anaconda.com/products/individual))
4. CUDA 10.1 설치 ([링크](https://developer.nvidia.com/cuda-10.1-download-archive-base))
5. cuDNN 7.6.0 설치 ([링크](https://developer.nvidia.com/rdp/cudnn-download))  
회원가입 후 cuDNN v7.6.0 for CUDA 10.1 설치

### CUDA 설정
1-5에서 다운받은 폴더의 아래 3개 파일들을 오른쪽 경로로 이동
- cuda/bin/cudnn64_7.dll -> C/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin
- cuda/include/cudnn.h -> C/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include
- cuda/cudnn/lib/x64/cudnn.lib -> C/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64

### 가상환경 생성 및 tesoflow-gpu 설치
Tensorflow gpu를 설치할 가상환경(tf2.0-gpu)을 생성하고, 가상환경 내에 tesoflow-gpu 설치
```
C:\Users\jieun>conda create --name tf2.0-gpu
```
```
C:\Users\jieun>conda activate tf2.0-gpu
(tf2.0-gpu) C:\Users\jieun> pip install tensorflow-gpu==2.2.0
```
