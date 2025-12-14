
# 🌱 MS AI School 3기 Team Cereal 

## 프로젝트 목표

- **자세 교정**: 운동 시 틀린 자세를 바로잡아 상해 위험을 줄입니다.
- **실시간 피드백**: AI가 운동자의 자세를 실시간으로 분석하여 피드백을 제공합니다.
- **사용자 경험 최적화**: 모바일과 웹 환경에서 최적의 사용자 경험을 제공합니다.

## 프로젝트 개요

| 단계 | 설명 | 주요 활동 | 결과물 | Project Repository |
|---|---|---|---|---|
| **1️⃣ Keypoint Detection 모델 개발** : <br><br>Real-time Exercise Posture Analysis with Keypoint Detection | 이 프로젝트는 운동 시 자세 교정을 위한 Keypoint Detection을 활용하여 사용자의 운동 자세를 실시간으로 분석하는 기능을 중점으로 합니다.  | - 실시간 키포인트 감지 기능 구현 (YOLOv8, AI Hub 데이터셋, 추가 데이터셋 구축) <br> - Gradio를 통한 웹 기반 프로토타입 개발 | - Github Pages에 iframe으로 넣어진 Gradio 웹 프로토타입 <br> - 실시간 자세 감지 및 분석 제공 <br> - Keypoint Detection Project 발표 PPT <br> - Github Repo (모델 공유 어려울시 HF Hub 고려) | [AI_Fitness_Trainer](https://github.com/msai-cereal/AI_Fitness_Trainer) |
| **2️⃣ LLM 통합을 통한 AI fitness trainer prototype 개발** : <br><br>Multimodal AI Fitness Trainer with Keypoint Detection and Language Models | 이 프로젝트는 Keypoint Detection과 자연어 처리 기술을 통합하여, 운동 자세 교정 피드백을 자연어로 제공하는 기능을 중점으로 합니다. | - 자연어 피드백 시스템 연동 (OpenAI API, LangChain) <br> - 사용자 경험(UX) 및 사용자 인터페이스(UI) 디자인 개선 <br> - 모바일 환경 최적화 <br> - 실 사용자 테스팅 및 피드백 반영 | - 휴대폰에서 최적화된 UX/UI를 갖춘 앱 형태의 서비스 <br> - 실시간 자세 교정 및 텍스트 기반 피드백 제공 <br> - Github Repo <br> - AI Project 발표 PPT |  |

---

## 🛠 팀원 & 역할
- **A(최아리)** - 모델 서빙 & SW 개발
- **B(김은교)** - 통합 시스템 엔지니어링 
- **C(김요셉)** - YOLOv8 모델링 
- **D(고정빈)** - 운동별 자세 평가 알고리즘 개발
- **E(박현상)** - 데이터 엔지니어링 
- **F([임태하](https://github.com/taehallm))** - 시퀀스 모델링 (LSTM)

## 발표자료
[MSAI3기_3팀_CEREAL_AI-Fitness-Trainer_발표자료.pdf](https://github.com/msai-cereal/.github/files/13371693/MSAI3._3._CEREAL_AI-Fitness-Trainer_.pdf)


## 🧙 감사의 말
- Shout out to AI Hub Korea!

---
Made with ❤️ by Team Cereal

