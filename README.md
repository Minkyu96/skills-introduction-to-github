import os
import json
import re
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoTokenizer, BertModel
from collections import defaultdict

import spacy
from flask import Flask, render_template, request, jsonify
from pyngrok import conf, ngrok

# ─────────────────────────────────────────────────────────────────────────────
# 1) ngrok 설정
# ─────────────────────────────────────────────────────────────────────────────
conf.get_default().ngrok_path = r"C:\Users\user\PycharmProjects\PythonProject7\ngrok.exe"
conf.get_default().auth_token = "2xi76Rsj0Qq7OynIy3atEmAApxb_6AE55zQ7AydPEwnREuegd"

# ─────────────────────────────────────────────────────────────────────────────
# 2) 응답 매핑 테이블 정의
# ─────────────────────────────────────────────────────────────────────────────
response_map = {
    # ===== 항공 =====
    ("항공", "수하물",   "조회"): "항공권 수하물 정보 페이지로 이동",
    ("항공", "금액",     "확인"): "항공권 금액 조회 페이지로 이동",
    ("항공", "금액",     "결제"): "항공권 결제 페이지로 이동",
    ("항공", "일정",     "조회"): "항공권 일정 조회 페이지로 이동",
    ("항공", "일정",     "변경"): "항공권 일정 변경 페이지로 이동",
    ("항공", "일정",     "취소"): "항공권 일정 취소 페이지로 이동",
    ("항공", "일정",     "예약"): "항공권 예매 페이지로 이동",
    ("항공", "환불",     "조회"): "항공권 환불 조회 페이지로 이동",
    ("항공", "환불",     "확인"): "항공권 환불 확인 페이지로 이동",
    ("항공", "환불",     "취소"): "항공권 환불 취소 페이지로 이동",
    ("항공", "예약상태", "확인"): "항공권 예약 상태 확인 페이지로 이동",

    # ===== 호텔 =====
    ("호텔", "수하물",   "조회"): "호텔 수하물 보관 서비스 안내 페이지로 이동",
    ("호텔", "금액",     "확인"): "호텔 요금 조회 페이지로 이동",
    ("호텔", "금액",     "결제"): "호텔 결제 페이지로 이동",
    ("호텔", "환불",     "조회"): "호텔 환불 조회 페이지로 이동",
    ("호텔", "환불",     "확인"): "호텔 환불 확인 페이지로 이동",
    ("호텔", "예약상태", "확인"): "호텔 예약 상태 확인 페이지로 이동",
    ("호텔", "일정",     "취소"): "호텔 예약 취소 페이지로 이동",
    ("호텔", "일정",     "예약"): "호텔 예약 페이지로 이동",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3) spaCy 한국어 모델 + EntityRuler 설정 (도시 / 날짜 / 시간 추출)
# ─────────────────────────────────────────────────────────────────────────────
nlp = spacy.load("ko_core_news_sm")

# 커스텀 패턴: “14시” 같은 TIME, “2025-06-02” 같은 DATE
patterns = [
    {"label": "TIME", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}시"}}]},
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{4}-\d{1,2}-\d{1,2}"}}]}
]
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(patterns)

# ─────────────────────────────────────────────────────────────────────────────
# 4) 한글 숫자 → 정수 변환 함수 (인원 수 계산에 사용)
# ─────────────────────────────────────────────────────────────────────────────
def korean_to_int(kor_word: str) -> int:
    units = {
        '한': 1, '두': 2, '세': 3, '네': 4, '다섯': 5,
        '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9
    }
    tens = {
        '열': 10, '스물': 20, '서른': 30, '마흔': 40, '쉰': 50,
        '예순': 60, '일흔': 70, '여든': 80, '아흔': 90
    }

    # “스물두” 처럼 접두사(열, 스물, 서른 …) + 단위(한, 두, 세 …)
    for t_word, t_val in tens.items():
        if kor_word.startswith(t_word):
            total = t_val
            remainder = kor_word[len(t_word):]
            if remainder:
                total += units.get(remainder, 0)
            return total

    # 단독으로 “한”, “두”, …, “열”
    return units.get(kor_word, 0)

def count_people(text: str) -> int:
    """
    문장 내에서 인원 수를 모두 합산하여 반환.
    - 아라비아 숫자 + 명 (예: "5명")
    - 복합 한글 숫자(1~99) + 명 (예: "한명", "열한명", "스물두명")
    """
    total = 0

    # 1) 아라비아 숫자 + “명”
    pattern_digits = re.compile(r'(\d+)\s*명')
    for match in pattern_digits.findall(text):
        total += int(match)

    # 2) 한글 숫자 (1~99) + “명”
    pattern_korean_all = re.compile(
        r'('
            r'(?:열|스물|서른|마흔|쉰|예순|일흔|여든|아흔)(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉)?'
            r'|'
            r'(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)'
        r')\s*명'
    )
    for match in pattern_korean_all.findall(text):
        total += korean_to_int(match)

    return total

# ─────────────────────────────────────────────────────────────────────────────
# 5) TriHead 모델 정의 + 토크나이저 & 학습된 가중치 로드
# ─────────────────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_DIR = r"C:\Users\user\Desktop\model_greeting"

class TriHead(nn.Module):
    def __init__(self, n_dom, n_cat, n_act, bert_name="klue/bert-base"):
        super().__init__()
        cfg = AutoConfig.from_pretrained(bert_name)
        self.bert = BertModel.from_pretrained(bert_name, config=cfg)
        self.drop = nn.Dropout(cfg.hidden_dropout_prob)
        self.dom  = nn.Linear(cfg.hidden_size, n_dom)
        self.cat  = nn.Linear(cfg.hidden_size, n_cat)
        self.act  = nn.Linear(cfg.hidden_size, n_act)

    def forward(self, ids, mask):
        pooled = self.bert(ids, attention_mask=mask)[1]  # [CLS] 토큰의 출력
        pooled = self.drop(pooled)
        return self.dom(pooled), self.cat(pooled), self.act(pooled)

# 라벨 맵 불러오기
with open(os.path.join(LOAD_DIR, "label_maps.json"), encoding="utf-8") as f:
    lbl = json.load(f)
dom2id = lbl["dom2id"]
cat2id = lbl["cat2id"]
act2id = lbl["act2id"]
id2dom = {int(v): k for k, v in dom2id.items()}
id2cat = {int(v): k for k, v in cat2id.items()}
id2act = {int(v): k for k, v in act2id.items()}

# 토크나이저 & 모델 인스턴스 생성
tok = AutoTokenizer.from_pretrained(LOAD_DIR)
model = TriHead(
    n_dom = len(dom2id),
    n_cat = len(cat2id),
    n_act = len(act2id),
    bert_name = "klue/bert-base"
).to(DEVICE)

# 학습된 가중치 로드
state_dict = torch.load(os.path.join(LOAD_DIR, "pytorch_model.bin"), map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

def predict_with_confidence(sentence: str) -> dict:
    enc = tok(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(DEVICE)

    with torch.no_grad():
        od, oc, oa = model(enc.input_ids, enc.attention_mask)

    dom_probs = F.softmax(od, dim=1)[0]  # (n_dom,)
    cat_probs = F.softmax(oc, dim=1)[0]  # (n_cat,)
    act_probs = F.softmax(oa, dim=1)[0]  # (n_act,)

    d_idx = int(dom_probs.argmax())
    c_idx = int(cat_probs.argmax())
    a_idx = int(act_probs.argmax())

    return {
        "domain":   (id2dom[d_idx], dom_probs[d_idx].item()),
        "category": (id2cat[c_idx], cat_probs[c_idx].item()),
        "action":   (id2act[a_idx], act_probs[a_idx].item())
    }

# 신뢰도 임계값 (예: 98%)
TRUTH_THRESHOLD = 0.98

# ─────────────────────────────────────────────────────────────────────────────
# 6) Flask 애플리케이션 정의
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route('/')
def home():
    """
    index.html을 렌더링합니다.
    (templates/index.html 내부에 간단한 채팅 UI를 구현해야 합니다.)
    """
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """
    클라이언트에서 JSON { "message": "<유저 입력>" } 형태로 요청을 보내면,
    모델 예측 → 신뢰도 체크 → 분기 로직 → reply 생성 → JSON으로 반환합니다.
    """

    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify(error="메시지를 입력해주세요."), 400

    # 1) 예측 + 확률 획득
    res       = predict_with_confidence(user_msg)
    dom, dom_conf = res["domain"]
    cat, cat_conf = res["category"]
    act, act_conf = res["action"]

    # 2) 신뢰도 체크: 어느 하나라도 기준 미만 시 “재입력 요청”
    if dom_conf < TRUTH_THRESHOLD or cat_conf < TRUTH_THRESHOLD or act_conf < TRUTH_THRESHOLD:
        return jsonify(
            user            = user_msg,
            domain          = {"label": dom, "conf": f"{dom_conf*100:.1f}%"},
            category        = {"label": cat, "conf": f"{cat_conf*100:.1f}%"},
            action          = {"label": act, "conf": f"{act_conf*100:.1f}%"},
            reply           = "❓ 죄송합니다. 입력하신 문장을 이해하지 못했습니다. 다시 입력해 주세요."
        )

    # 3) “인사” 액션 처리
    if act == "인사":
        return jsonify(
            user            = user_msg,
            domain          = {"label": dom, "conf": f"{dom_conf*100:.1f}%"},
            category        = {"label": cat, "conf": f"{cat_conf*100:.1f}%"},
            action          = {"label": act, "conf": f"{act_conf*100:.1f}%"},
            reply           = "넹 안녕하세요! 어떻게 도와드릴까요?"
        )

    # 4) 도메인 Unknown & 카테고리 일정 & 액션 예약 → “항공/호텔 선택 요청”
    if dom == "Unknown" and cat == "일정" and act == "예약":
        return jsonify(
            user            = user_msg,
            domain          = {"label": dom, "conf": f"{dom_conf*100:.1f}%"},
            category        = {"label": cat, "conf": f"{cat_conf*100:.1f}%"},
            action          = {"label": act, "conf": f"{act_conf*100:.1f}%"},
            reply           = "어떤 것을 예약해드릴까요? (항공 / 호텔 중 선택해주세요)"
        )

    # 5) “항공 예약” 처리: 도시, 날짜, 시간, 인원 수 추출
    if dom == "항공" and cat == "일정" and act == "예약":
        doc    = nlp(user_msg)
        city   = None
        date   = None
        time_  = None
        people = count_people(user_msg)

        for ent in doc.ents:
            if ent.label_ in ("LOC", "LC"):
                city = ent.text
            elif ent.label_ == "DT":
                date = ent.text
            elif ent.label_ == "TIME":
                time_ = ent.text

        reply_parts = ["항공 예약을 도와드릴게요."]
        if city:
            reply_parts.append(f"출발/도착 도시: {city}")
        else:
            reply_parts.append("도시 정보가 없습니다.")
        if date:
            reply_parts.append(f"날짜: {date}")
        else:
            reply_parts.append("날짜 정보가 없습니다.")
        if time_:
            reply_parts.append(f"시간: {time_}")
        else:
            reply_parts.append("시간 정보가 없습니다.")
        if people and people > 0:
            reply_parts.append(f"인원 수: {people}명")
        else:
            reply_parts.append("인원 수 정보가 없습니다.")
        reply_text = " ㆍ".join(reply_parts)

        return jsonify(
            user            = user_msg,
            domain          = {"label": dom, "conf": f"{dom_conf*100:.1f}%"},
            category        = {"label": cat, "conf": f"{cat_conf*100:.1f}%"},
            action          = {"label": act, "conf": f"{act_conf*100:.1f}%"},
            reply           = reply_text
        )

    # 6) “호텔 예약” 처리: 도시, 체크인 날짜, 인원 수 추출
    if dom == "호텔" and cat == "일정" and act == "예약":
        doc    = nlp(user_msg)
        city   = None
        date   = None
        people = count_people(user_msg)

        for ent in doc.ents:
            if ent.label_ in ("LOC", "LC"):
                city = ent.text
            elif ent.label_ == "DT":
                date = ent.text

        reply_parts = ["호텔 예약을 도와드릴게요."]
        if city:
            reply_parts.append(f"지역(도시): {city}")
        else:
            reply_parts.append("지역 정보가 없습니다.")
        if date:
            reply_parts.append(f"체크인 날짜: {date}")
        else:
            reply_parts.append("체크인 날짜 정보가 없습니다.")
        if people and people > 0:
            reply_parts.append(f"인원 수: {people}명")
        else:
            reply_parts.append("인원 수 정보가 없습니다.")
        reply_text = " ㆍ".join(reply_parts)

        return jsonify(
            user            = user_msg,
            domain          = {"label": dom, "conf": f"{dom_conf*100:.1f}%"},
            category        = {"label": cat, "conf": f"{cat_conf*100:.1f}%"},
            action          = {"label": act, "conf": f"{act_conf*100:.1f}%"},
            reply           = reply_text
        )

    # 7) “통합(항공+호텔) 예약” 처리: 도시, 날짜, 시간, 인원 추출
    if dom == "통합" and cat == "일정" and act == "예약":
        doc    = nlp(user_msg)
        city   = None
        date   = None
        time_  = None
        people = count_people(user_msg)

        for ent in doc.ents:
            if ent.label_ in ("LOC", "LC"):
                city = ent.text
            elif ent.label_ == "DT":
                date = ent.text
            elif ent.label_ == "TIME":
                time_ = ent.text

        reply_parts = ["통합(항공+호텔) 예약을 도와드릴게요."]
        if city:
            reply_parts.append(f"지역(도시): {city}")
        else:
            reply_parts.append("지역 정보가 없습니다.")
        if date:
            reply_parts.append(f"날짜: {date}")
        else:
            reply_parts.append("날짜 정보가 없습니다.")
        if time_:
            reply_parts.append(f"시간: {time_}")
        else:
            reply_parts.append("시간 정보가 없습니다.")
        if people and people > 0:
            reply_parts.append(f"인원 수: {people}명")
        else:
            reply_parts.append("인원 수 정보가 없습니다.")
        reply_text = " ㆍ".join(reply_parts)

        return jsonify(
            user            = user_msg,
            domain          = {"label": dom, "conf": f"{dom_conf*100:.1f}%"},
            category        = {"label": cat, "conf": f"{cat_conf*100:.1f}%"},
            action          = {"label": act, "conf": f"{act_conf*100:.1f}%"},
            reply           = reply_text
        )

    # 8) 그 외: 미리 정의된 response_map에서 응답 가져오기
    key     = (dom, cat, act)
    reply   = response_map.get(key, "죄송합니다. 해당 요청을 처리할 수 없습니다.")

    return jsonify(
        user            = user_msg,
        domain          = {"label": dom, "conf": f"{dom_conf*100:.1f}%"},
        category        = {"label": cat, "conf": f"{cat_conf*100:.1f}%"},
        action          = {"label": act, "conf": f"{act_conf*100:.1f}%"},
        reply           = reply
    )


if __name__ == "__main__":
    # ngrok 터널 열기
    public_url = ngrok.connect(5000).public_url
    print(" * ngrok tunnel:", public_url)

    # Flask 서버 실행
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
