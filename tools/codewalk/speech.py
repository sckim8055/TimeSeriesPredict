"""음성 대본 도구 — 화면에 보이는 문장(say, 숫자는 아라비아 숫자)을 소리 내어 읽을 문장(speech)으로 바꾼다.

원칙
- 숫자는 음성 엔진이 읽는 방식에 맡기지 않고 한글로 풀어 쓴다(0.855 → "영 점 팔오오", 2023-12 → "이천이십삼 년 십이 월").
  같은 숫자가 엔진·음성마다 다르게 읽히는 것을 막고, 화면 숫자와 소리가 같은지 사람이 대본만 보고 확인할 수 있게 한다.
- 문장(자막 한 줄)마다 따로 바꾼다 → 자막 줄 i 와 음성 구간 i 가 1:1 로 맞는다(segments).
- 영어 약어는 한국어 발음으로(RMSE → 알엠에스이). 남은 영문·코드 이름은 leftovers() 로 찾아 사람이 고친다.
시험: tests/test_speech.py
"""
from __future__ import annotations

import html
import re
import unicodedata

DIG = "영일이삼사오육칠팔구"
_NAT1 = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NAT10 = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]


def sino(n: int) -> str:
    """한자어 수: 120 → 백이십, 2023 → 이천이십삼, 10000 → 만, 3116 → 삼천백십육."""
    n = int(n)
    if n < 0:
        return "마이너스 " + sino(-n)
    if n == 0:
        return "영"

    def four(m):
        s = ""
        for unit, val in (("천", 1000), ("백", 100), ("십", 10), ("", 1)):
            d = m // val % 10
            if d:
                s += ("" if d == 1 and unit else DIG[d]) + unit
        return s
    parts = []
    for big, div in (("조", 10 ** 12), ("억", 10 ** 8), ("만", 10 ** 4), ("", 1)):
        g = n // div % 10000
        if g:
            parts.append(("" if g == 1 and big == "만" else four(g)) + big)
    return " ".join(parts)


def native(n: int) -> str:
    """고유어 수(관형형, 1~99): 1 → 한, 20 → 스무, 21 → 스물한. 그 밖은 한자어."""
    n = int(n)
    if not 1 <= n <= 99:
        return sino(n)
    if n == 20:
        return "스무"
    return _NAT10[n // 10] + _NAT1[n % 10]


def cnt(n, unit: str) -> str:
    """고유어로 세는 단위(개·번·가지·그루·명·칸·줄·자리): cnt(20, '개') → '스무 개'. 100 이상은 한자어."""
    return f"{native(n)} {unit}"


def numstr(s: str) -> str:
    """화면에 적힌 숫자 문자열 그대로 읽기: '0.855' → '영 점 팔오오', '1,234' → '천이백삼십사', '-0.42' → '마이너스 영 점 사이'."""
    s = s.replace(",", "").replace("−", "-").replace("+", "")
    neg = s.startswith("-")
    s = s.lstrip("-")
    if "." in s:
        a, b = s.split(".", 1)
        out = f"{sino(int(a or 0))} 점 " + "".join(DIG[int(c)] for c in b)
    else:
        out = sino(int(s))
    return ("마이너스 " if neg else "") + out


def _date(m: re.Match, with_day: bool = False) -> str:
    """2023-12-01 → 이천이십삼 년 십이 월 (1일은 월 단위 자료의 날짜라 읽지 않음), 2023-12-15 → … 십오 일."""
    y, mo = int(m.group(1)), int(m.group(2))
    d = int(m.group(3)) if with_day else None
    mo_s = {6: "유월", 10: "시월"}.get(mo, f"{sino(mo)} 월")                 # 6월·10월은 유월·시월로 읽는다
    return f"{sino(y)} 년 {mo_s}" + (f" {sino(d)} 일" if d and d != 1 else "")


# 한국어 발음 (긴 것부터 바꾼다 — 단어 경계에서만)
TERMS = {
    "LightGBM": "라이트지비엠", "SARIMA": "사리마", "ARIMA": "아리마", "VECM": "브이이씨엠", "LSTM": "엘에스티엠", "GRU": "지알유", "BIVA": "비바",
    "BRITS": "브리츠", "VAE": "브이에이이", "Ridge": "릿지", "RMSE": "알엠에스이", "MAE": "엠에이이", "MSE": "엠에스이", "RSI": "알에스아이", "MACD": "엠에이씨디",
    "EMA": "지수이동평균", "DM": "디엠", "DSR": "디에스알", "AIC": "에이아이씨", "BIC": "비아이씨", "OLS": "오엘에스", "SHAP": "샵", "ReLU": "렐루",
    "naive": "나이브", "sklearn": "사이킷런", "numpy": "넘파이", "pandas": "판다스", "PyTorch": "파이토치", "torch": "토치", "statsmodels": "스탯츠모델스",
    "alpha": "알파", "embargo": "엠바고", "ffill": "앞 값 채우기", "lag": "래그", "epoch": "에포크", "batch": "배치", "Adam": "아담",
    "B&H": "바이 앤 홀드", "KODEX": "코덱스", "KB": "케이비", "KOSPI": "코스피", "SOX": "삭스", "VIX": "빅스", "USDKRW": "원 달러 환율",
    "F11": "에프 십일", "F12": "에프 십이", "R²": "알 제곱", "R2": "알 제곱", "pct": "변화율", "diff": "차이",
    "LSTMCell": "엘에스티엠 셀", "RITS": "리츠", "RIN": "린", "KLD": "케이엘디", "CPU": "씨피유", "GPU": "지피유", "MinMax": "민맥스", "EarlyStopping": "얼리 스토핑",
    "GRUCell": "지알유 셀", "MLE": "최우추정", "ADF": "에이디에프", "KPSS": "케이피에스에스", "Johansen": "요한센", "MAPE": "엠에이피이", "SMA": "단순이동평균",
    "CAGR": "연평균 성장률", "MDD": "최대 낙폭", "Sharpe": "샤프", "Kalman": "칼만", "AR": "에이알", "MA": "엠에이", "tanh": "하이퍼볼릭 탄젠트",
    "sigmoid": "시그모이드", "dropout": "드롭아웃", "Dropout": "드롭아웃", "SK": "에스케이", "ARMA": "자기회귀 이동평균", "L-BFGS": "엘비에프지에스", "SARIMAX": "사리맥스", "Dense": "덴스", "Linear": "리니어", "StandardScaler": "표준화",
}
_LETTER = {c: n for c, n in zip("abcdefghijklmnopqrstuvwxyz", ["에이", "비", "씨", "디", "이", "에프", "지", "에이치", "아이", "제이", "케이", "엘", "엠", "엔", "오", "피", "큐", "알",
                                                              "에스", "티", "유", "브이", "더블유", "엑스", "와이", "제트"])}
OPS = [("×", " 곱하기 "), ("÷", " 나누기 "), ("≈", " 약 "), ("≤", " 이하 "), ("≥", " 이상 "), ("→", ", "), ("·", ", "), ("—", ", "), ("–", ", "), ("~", " 에서 "),
       ("⊙", " 원소별 곱 "), ("√", "루트 "), ("%", " 퍼센트"), ("①", "첫째, "), ("②", "둘째, "), ("③", "셋째, ")]
NAMES = [("σ²", "시그마 제곱"), ("log σ²", "로그 시그마 제곱"), ("Σ", "합"), ("σ", "시그마"), ("μ", "뮤"), ("α", "알파"), ("β", "베타"), ("γ", "감마"), ("δ", "델타"), ("θ", "세타"),
         ("φ", "파이"), ("Φ", "대문자 파이"), ("Δ", "델타 "), ("η", "에타"), ("ρ", "로우"), ("ω", "오메가"), ("Π", "대문자 파이"), ("Γ", "대문자 감마"), ("Θ", "대문자 세타"), ("ε", "엡실론"), ("λ", "람다"), ("τ", "타우"), ("ŷ", "와이 햇"), ("x̂", "엑스 햇"), ("ẑ", "제트 햇"), ("ĉ", "씨 햇"), ("½", "이분의 일")]
SYM = OPS + NAMES   # (예전 이름 — 시험·다른 코드 호환)


_JOSA = {"은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"), "가": ("이", "가"), "을": ("을", "를"), "를": ("을", "를"), "과": ("과", "와"), "와": ("과", "와")}
J = r"(?:(?P<j>으로|은|는|이|가|을|를|과|와|로)(?=[\s,.?!:;)\]]|$))?"


def _batchim(word: str) -> int:
    """마지막 한글 글자의 받침 번호(0 = 없음, 8 = ㄹ). 한글이 아니면 0."""
    c = word.rstrip()[-1:] if word.strip() else ""
    return (ord(c) - 0xAC00) % 28 if c and 0xAC00 <= ord(c) <= 0xD7A3 else 0


def attach(word: str, j: str | None) -> str:
    """바뀐 낱말 뒤 조사를 받침에 맞게: attach('알 제곱', '는') → '알 제곱은', attach('십', '로') → '십으로'."""
    if not j:
        return word
    b = _batchim(word)
    if j in ("으로", "로"):
        return word + ("으로" if b and b != 8 else "로")
    yes, no = _JOSA[j]
    return word + (yes if b else no)


def _sub(pattern: str, fn, t: str) -> str:
    """pattern 을 fn 으로 바꾸고 바로 뒤에 붙은 조사를 고친다."""
    return re.sub(pattern + J, lambda m: attach(fn(m), m.group("j")), t)


def speakify(text: str) -> str:
    """화면 문장 → 읽을 문장. 태그 제거 → (코드 이름) 괄호 제거 → 날짜 → 지수 표기 → 숫자 → 약어 → 기호 → 한 글자 영문. 바뀐 낱말 뒤 조사는 받침에 맞춘다."""
    t = html.unescape(re.sub(r"<[^>]+>", "", text or ""))
    t = _sub(r"([^\s(]*)\(([A-Za-z][A-Za-z0-9_]*)\)", lambda m: m.group(1), t)   # '6개월 RSI(A_rsi_6)' 의 (A_rsi_6) — 화면용 코드 이름(조사는 앞 낱말에 맞춤)
    t = re.sub(r"([α-ωΑ-Ω])(?=\d)", r"\1 ", t)                              # τ6 → τ 6 → 타우 육, φ1 → 파이 일
    t = _sub(r"(?<![0-9])(\d{4})-(\d{2})-(\d{2})(?![0-9])", lambda m: _date(m, True), t)
    t = _sub(r"(?<![0-9])(\d{4})-(\d{2})(?![0-9-])", _date, t)
    t = _sub(r"([+−-]?\d+(?:\.\d+)?)e([+−-]?\d+)", lambda m: f"{numstr(m.group(1))} 곱하기 십의 {numstr(m.group(2))} 제곱", t)
    t = re.sub(r"(?<![A-Za-z0-9_.])(\d{1,2})\s*(개(?!월)|번|가지|그루|명|칸|줄(?!었|어|여|인|일)|자리)", lambda m: cnt(int(m.group(1)), m.group(2)), t)
    t = _sub(r"(?<![A-Za-z0-9_.])([+−-]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![A-Za-z0-9_])",
             lambda m: numstr((m.group(1) if m.group(1) and m.group(1) in "−-" else "") + m.group(2) + (m.group(3) or "")), t)
    for k in sorted(TERMS, key=len, reverse=True):
        t = _sub(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z0-9])", lambda m, k=k: TERMS[k], t)
    for a, b in sorted(NAMES, key=lambda x: -len(x[0])):
        t = _sub(re.escape(a), lambda m, b=b: b, t)                            # 이름 뒤 조사는 받침에 맞춤(x̂가 → 엑스 햇이)
    for a, b in OPS:
        t = t.replace(a, b)
    t = re.sub(r"(?<=[가-힣A-Za-z])_(?=[A-Za-z가-힣])", " ", t)                # γ_h → 감마 h
    t = re.sub(r"(\S+)\s*=\s*", lambda m: attach(m.group(1), "은") + " ", t)
    t = t.replace("−", " 빼기 ").replace("+", " 더하기 ")
    t = _sub(r"(?<![A-Za-z0-9_])([A-Za-z])(\d+)(?![A-Za-z0-9_])", lambda m: _LETTER[m.group(1).lower()] + " " + sino(int(m.group(2))), t)   # L2 → 엘 이
    t = _sub(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z0-9_])", lambda m: _LETTER[m.group(1).lower()], t)
    t = re.sub(r"[()\[\]'\"]", " ", t)
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r",\s*,", ",", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def leftovers(speech: str) -> list[str]:
    """읽을 문장에 남은 영문·숫자·그리스 문자·수학 기호(사람이 고쳐야 할 것 — 음성 엔진이 엉뚱하게 읽거나 건너뛴다)."""
    return sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_./]*|\d+|[\u0370-\u03ff\u2200-\u22ff\u2190-\u21ff×÷±≈≤≥√∑∏∂∇]", speech)))


def segments(say: str, max_len: int = 70) -> list[str]:
    """자막 줄로 나누기: 문장 끝(. ? !) 마다, 그래도 길면 쉼표·줄표에서. 소수점·날짜는 나누지 않는다."""
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", say or ""))).strip()
    out = []
    for s in re.split(r"(?<=[.?!])\s+(?=\S)", t):
        while len(s) > max_len:
            cut = max((m.end() for m in re.finditer(r"[,:;] | — |\) ", s[: max_len + 10]) if m.end() >= 15), default=0)
            if not cut:
                break
            out.append(s[:cut].strip()); s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def _hangul(c: str) -> bool:
    return bool(c) and 0xAC00 <= ord(c) <= 0xD7A3


_JOSA_DISP = re.compile(r"(\S+?)(\s?)(으로|은|는|이|가|을|를|과|와|로)(?=[\s,.?!:;)\]<]|$)")
_TAG = re.compile(r"<[^>]+>")
_OPS_SPLIT = re.compile(r"[·→←=+−×⊙*]")


def _symbolic(head: str) -> bool:
    """읽는 소리가 정해진 기호·숫자로 끝나는 낱말인지: 숫자·%·그리스 문자·위첨자·결합 기호(x̂ 의 ^)·한 글자 라틴 문자(+숫자)(h, x̂, N6, L2, τ6).
    여러 글자 영어 낱말(epoch, MSE, weight_combine)은 사람마다 읽는 법이 달라 손대지 않는다."""
    c = head[-1]
    cat = unicodedata.category(c)
    if c.isdigit() or c in "%²³¹⁰⁴⁵⁶⁷⁸⁹" or cat[0] == "M" or "GREEK" in unicodedata.name(c, ""):
        return True
    core = re.sub(r"[\u0300-\u036f0-9]", "", unicodedata.normalize("NFD", head))
    return len(core) == 1 and core.isalpha()


def josa_display(text: str) -> str:
    """화면 글(자막·화면 설명)의 조사를 읽는 소리에 맞춘다: '81.3104을' → '81.3104를'(사), '3는' → '3은'(삼), 'N6는' → 'N6은'(엔 육), 'x̂ 로' → 'x̂ 으로'(햇).
    규칙(한국어 맞춤법): 괄호 뒤 조사는 괄호 앞 낱말에 맞춘다('창(이 폴드 59개)을' 은 '창을'). 연산 기호로 이어진 말은 마지막 항에 맞춘다('x̂·ẑ' → ẑ).
    고치는 것은 조사 앞 낱말이 숫자·기호·한 글자 문자로 끝날 때뿐이다(한글·여러 글자 영어 낱말·분수 'a/b' 는 그대로). HTML 태그는 빼고 본다."""
    def head_of(m) -> str:
        word = m.group(1)
        plain = _TAG.sub("", word)
        if plain.endswith(")") or plain.endswith("]"):       # 괄호 앞 낱말 찾기(괄호 속에 띄어쓰기가 있어도)
            close, open_ = plain[-1], "(" if plain[-1] == ")" else "["
            depth, k = 0, m.end(1) - 1
            while k >= 0:
                ch = text[k]
                if ch == close:
                    depth += 1
                elif ch == open_:
                    depth -= 1
                    if depth == 0:
                        break
                k -= 1
            if k < 0:
                return ""
            if close == "]":                                   # [3, 11, 34] 같은 모양·목록은 마지막 항을 읽는다
                inner = _TAG.sub("", text[k + 1:m.end(1) - 1])
                items = [x.strip() for x in inner.split(",") if x.strip()]
                return items[-1] if items else ""
            pre = re.sub(r"(?:<[^>]+>|\s)+$", "", text[:k])
            plain = _TAG.sub("", pre.split()[-1]) if pre.split() else ""
            plain = re.sub(r"^.*[(\[]", "", plain)
        return plain

    def fix(m):
        gap, j = m.group(2), m.group(3)
        w = _TAG.sub("", m.group(1))
        if not w or _hangul(w[-1]):                    # 한글 낱말 끝(경로·사이·없는)은 조사가 아니거나 이미 맞음
            return m.group(0)
        if not (unicodedata.category(w[-1])[0] in "LNM" or w[-1] in ")]}%"):   # 쉼표·마침표 뒤의 '이'(이 값)는 조사가 아님
            return m.group(0)
        head = head_of(m)
        if not head or "/" in head:
            return m.group(0)
        if _hangul(head[-1]):
            spoken = head
        else:
            seg = [x for x in _OPS_SPLIT.split(head) if x.strip()]
            head = seg[-1].strip() if seg else ""
            if not head or _hangul(head[-1]):
                spoken = head
            elif _symbolic(head):
                spoken = re.sub(r"[^\w]+$", "", speakify(head) or "")
            else:
                return m.group(0)
        if not spoken or not _hangul(spoken[-1]):
            return m.group(0)
        return m.group(1) + gap + attach(spoken, j)[len(spoken):]
    return _JOSA_DISP.sub(fix, text)


def prepare(step: dict) -> dict:
    """대본 한 단계 → 화면 자막 줄(captions)과 읽을 문장(speech, 줄마다 1:1). say 가 없으면 narr 를 그대로 쓴다(say_auto=True — 사람이 다듬을 것)."""
    src = step.get("say") or step.get("narr") or ""
    cap = [josa_display(c) for c in segments(src)]
    return {"captions": cap, "speech": [speakify(c) for c in cap], "say_auto": not step.get("say")}
