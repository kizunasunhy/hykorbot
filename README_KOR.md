# hykorbot

## Usage
### 필요한 패키지
```
pytorch >= 0.4
pytorch ignite
MXNet == 1.5.0 or higher
gluonnlp == 0.8.1
```
## 학습
### 데이터 전처리 
preprocessing.py 실행합니다.

train_data가 이런 형식을 가져야 합니다: 
```
{
"utterances": [{"candidates": ["미세먼지 지수 높을 때는 실외말고 실내생활해야지.",
"메일 보내면 교수님 답장이 더 잘 오는 편이야, 조교님 답장이 더 잘오는 편이야?",
"비 올 때 빨래하면 그냥 두지 말고 건조기 사용하도록 해.",
"시작 날짜가 언제지? 내장산 단풍.",
"전화해. 급한 연락은 메시지 보내지 말고.",
"와디즈 광고메일이랑 텀블벅이랑 비교했을 때 뭐가 다른가요?",
"세척기말고 냉장고 안에 있는 내용물 어떻게 확인해?",
"서재 조명등 이단계로 할까요, 삼단계로 할까요? 당신은 무엇을 원하시나요?",
"안방 안에 있는 조명 있잖아요 그거 어떻게 켜요?",
"약속을 바꿀 수는 있지만 하루 전에는 바꾸는 거 아니니까 그렇게 하면 안돼.",
"친구랑 설날에 약속 잡으면 부모님한테 욕 먹으니까 잡지 마시죠.",
"해는 직접 처다보지 마라.",
"자외선이 강할 때 피부노출 하지 마세요.",
"네이트 메일을 사용할 때 전체를 대상으로 메일을 어떻게 보내?",
"벚꽃 축제로 유명한 도시는?",
"내가 쓴 메일을 가장 많이 받은 거래처는 어디야?“],
"history": ["내가 가장 많이 메일을 쓴 거래처를 알려줘“]}]
}
```
이런 utterances를 가지고 있는 dict를 구성됩니다. 
각 utterances에서 candidates가 16개 있고 history가 하나 있습니다. 
"history"의 문장과 "candidates의 마지막 문장이 유사 문장이라고 볼 수 있습니다. 

KoGPT2 model을 training할 때는 task가 두개 있습니다:

1. *Language modeling*
2. *Next-sentence prediction*

그 중에서 *Next-sentence prediction *임무에서 "history"의 *Next-sentence*를 예측합니다.  
그래서 요기서 "candidates"[-1] (즉: "내가 쓴 메일을 가장 많이 받은 거래처는 어디야?“)은 
"history": ["내가 가장 많이 메일을 쓴 거래처를 알려줘“]의 gold answer입니다. 
나머지("candidates"[0] ~ "candidates"[14])는 distractor입니다. 
Next-sentence prediction할 때 distractor도 필요합니다.
