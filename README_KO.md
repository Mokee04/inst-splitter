# inst-splitter (한국어 가이드)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

**5개 이종 AI 분리 모델 앙상블 기반 커버곡용 자동 반주(MR) 및 상호보완적 Vocals 생성기**

[English README](./README.md) | [이슈 / 버그 제보](https://github.com/Mokee04/inst-splitter/issues)

</div>

---

## 📌 소개

`inst-splitter`는 커버곡 제작 및 음원 분석을 위해 설계된 전문가용 AI 음원 분리 엔진입니다. 3개의 RoFormer 모델과 2개의 Demucs 모델을 **계층적 합의(Hierarchical Consensus)** 알고리즘으로 결합하여, 위상 손상이나 공간감 손실 없는 최상의 **MR(반주)**과 원본과 완벽히 합산되는 **상호보완적 Vocals**를 전자동으로 추출합니다.

---

## 🎯 핵심 기능

1. **5-Model Heterogeneous Ensemble (계층적 합의)**:
   - **RoFormer 계열 3종**: `Bleedless`, `Balanced`, `Fullness (Carrier)`
   - **Demucs 계열 2종**: `HTDemucs FT`, `Demucs MDX Extra`
   - 두 아키텍처 계열이 모두 보컬로 판단할 때만 강력 감쇠하며, 불일치 시 악기 질감과 리버브 테일을 100% 보존합니다.
2. **상호보완적 Vocals 동시 추출 (Complementary Residual)**:
   - 원키($0st$): $\text{Vocals} = \text{Original} - \text{MR}$
   - 피치 시프트($\pm Nst$): $\text{Vocals} = P(\text{Original}) - P(\text{MR})$
   - 32-bit Float 정밀도에서 $\text{MR} + \text{Vocals} = \text{Original}$ 보존 불변식 보장.
3. **Linked Peak Safety (연동 피크 보호)**:
   - MR과 Vocals의 최대 피크 중 큰 값을 기준으로 **단 하나의 공통 게인($g$)**을 적용하여 $-1.0\text{ dBFS}$ 이하로 제어 (두 스템 간 상대 레벨 왜곡 없음).
4. **비대칭 엔벨로프 스무딩 DSP**:
   - Attack (40ms) 빠른 감쇠 + Release (100ms) 완만한 복원으로 Chirping 및 Pumping 노이즈 원천 차단.
5. **원샷 트랜스포즈 (Rubber Band)**:
   - 반주 전체에 피치 시프트를 1회 적용하여 드럼 분리 왜곡 방지.
6. **24-bit Studio Master WAV 출력**:
   - 44.1kHz Stereo 24-bit Signed PCM 포맷.

---

## ⚡ 빠른 시작 (Quick Start)

### 1. 시스템 필수 도구 설치

- **macOS** (Homebrew):
  ```bash
  brew install ffmpeg rubberband libsndfile
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg rubberband-cli libsndfile1
  ```
- **Windows** (Scoop):
  ```powershell
  scoop install ffmpeg rubberband
  ```

### 2. 설치

```bash
git clone https://github.com/Mokee04/inst-splitter.git
cd inst-splitter
pip install -e .
```

### 3. 사용법

#### 단일 곡 분리 및 키 변환:
```bash
# 원키(0st)로 분리
inst-split "노래.flac" 0

# 5키 낮춤 (-5st)
inst-split "노래.flac" -5

# 2키 올림 (+2st)
inst-split "노래.mp3" +2
```

결과물은 음원 폴더 내 `MR_output/` 디렉터리에 2개의 파일로 저장됩니다:
- `{파일명}_MR_{-5st}.wav`
- `{파일명}_Vocals_{-5st}.wav`

#### 폴더 단위 일괄(배치) 변환:
```bash
# 폴더 내 모든 음원 파일을 -5키로 일괄 변환
inst-splitter-batch "/Music/Album" -5

# 하위 폴더까지 재귀 탐색하여 원키로 일괄 변환
inst-splitter-batch "/Music/Library" 0 --recursive --output-dir "./results"
```

---

## 📄 라이선스

본 프로젝트는 [MIT License](./LICENSE)에 따라 자유롭게 사용 및 수정이 가능합니다.
