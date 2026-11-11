"""음성 대본(tools/codewalk/speech.py)·녹음 시각(tools/narrate.py)·직접 실행해 보기(common.run_lab) 시험 (2026-11-07)."""
from __future__ import annotations

import pytest

from tools.codewalk import speech as sp


@pytest.mark.parametrize("s, want", [("0.855", "영 점 팔오오"), ("1,234", "천이백삼십사"), ("-0.42", "마이너스 영 점 사이"), ("−0.42", "마이너스 영 점 사이"),
                                     ("3116", "삼천백십육"), ("120", "백이십"), ("10000", "만"), ("2023", "이천이십삼"), ("0", "영"), ("+0.054", "영 점 영오사")])
def test_numstr(s, want):
    assert sp.numstr(s) == want


def test_native_counts():
    assert sp.cnt(20, "개") == "스무 개" and sp.cnt(21, "개") == "스물한 개" and sp.cnt(43, "개") == "마흔세 개" and sp.cnt(120, "개") == "백이십 개"


@pytest.mark.parametrize("t, want", [
    ("결정일 2023-12-01의 실제", "결정일 이천이십삼 년 십이 월의 실제"),               # 월 자료의 1일은 읽지 않음, 날짜 뒤 '의' 를 깨지 않음
    ("2024-11-19 ~ 2024-12-30", "이천이십사 년 십일 월 십구 일 에서 이천이십사 년 십이 월 삼십 일"),
    ("R²는 0.855입니다", "알 제곱은 영 점 팔오오입니다"),                             # 바뀐 낱말 뒤 조사를 받침에 맞춤
    ("알파는 10로 세로선", "알파는 십으로 세로선"),
    ("1.4e-17로 사실상 0입니다", "일 점 사 곱하기 십의 마이너스 십칠 제곱으로 사실상 영입니다"),
    ("6개월 RSI(A_rsi_6)는 z가 -1.88로,", "육개월 알에스아이는 제트가 마이너스 일 점 팔팔로,"),   # (코드 이름)은 읽지 않음
    ("1개월 수익률(A_ret_1)는", "일개월 수익률은"),
    ("120개 중 12개", "백이십개 중 열두 개"),
    ("예측 +0.054%, 실제 -0.123%", "예측 영 점 영오사 퍼센트, 실제 마이너스 영 점 일이삼 퍼센트"),
])
def test_speakify(t, want):
    assert sp.speakify(t) == want


def test_segments_and_prepare_are_one_to_one():
    st = {"narr": "x", "say": "첫 문장입니다. 값은 0.855입니다! 아주 긴 문장은, " + "가나다라마바사 " * 12 + "쉼표에서 나뉩니다."}
    p = sp.prepare(st)
    assert len(p["captions"]) == len(p["speech"]) >= 3 and not p["say_auto"]
    assert p["captions"][1] == "값은 0.855입니다!" and p["speech"][1] == "값은 영 점 팔오오입니다!"
    assert sp.prepare({"narr": "<b>굵게</b> 1개."})["say_auto"] is True


def test_seg_times_exact_and_fallback():
    from tools.narrate import seg_times
    segs = ["가나.", "다라마."]
    text = " ".join(segs)
    al = {"characters": list(text), "character_start_times_seconds": [0.1 * i for i in range(len(text))], "character_end_times_seconds": [0.1 * i + 0.1 for i in range(len(text))]}
    out, total, exact = seg_times(segs, al)
    assert exact and out == [[0.0, 0.3], [0.4, 0.8]] and total == 0.8
    al2 = {**al, "characters": list("다른 글자 들")}
    _, _, exact2 = seg_times(segs, al2)
    assert exact2 is False


def test_run_lab_checks_output_and_pure_imports():
    from tools.codewalk.common import run_lab
    ok = run_lab("x = [1.0, 2.0]\nprint(sum(x) / len(x))", "1.5", pure=True)
    assert ok["ok"] and ok["got"] is None
    bad = run_lab("print(2)", "1.5", pure=True)
    assert not bad["ok"] and bad["got"] == "2"
    with pytest.raises(AssertionError):
        run_lab("import numpy\nprint(1)", "1", pure=True)


@pytest.mark.parametrize("t, want", [
    ("값 81.3104을 81.1799로 나누고", "값 81.3104를 81.1799로 나누고"),     # 사(받침 없음) → 를
    ("81.1799은 1이 됩니다", "81.1799는 1이 됩니다"),
    ("111 나누기 3는 37이므로", "111 나누기 3은 37이므로"),               # 삼 → 은
    ("6가 짝수라", "6이 짝수라"),                                          # 육 → 이
    ("0.0813로 줄었습니다", "0.0813으로 줄었습니다"),                       # 삼 → 으로
    ("최대 차이는 0로,", "최대 차이는 0으로,"),                            # 영 → 으로
    ("알파는 10로", "알파는 10으로"),                                       # 십(ㅂ) → 으로
    ("1.4e-17로", "1.4e-17으로"),                                          # …제곱(ㅂ) → 으로
    ("N6는 2015-01부터", "N6은 2015-01부터"),                               # 엔 육 → 은
    ("RSI(A_rsi_6)는", "RSI(A_rsi_6)는"),                                  # 괄호 속 코드 이름은 읽지 않음 → 아이 → 는
    ("수익률(A_ret_1)는", "수익률(A_ret_1)은"),
    ("BRITS가 채웁니다", "BRITS가 채웁니다"),                                # 이미 맞으면 그대로
    ("학습 창 111개를", "학습 창 111개를"),                                  # 한글로 끝나는 낱말은 손대지 않음
    ("2020-03의 정답", "2020-03의 정답"),                                   # 조사 아닌 '의'
    ("3이면 됩니다", "3이면 됩니다"),                                        # 조사 뒤에 글자가 더 붙으면 손대지 않음
])
def test_josa_display_matches_spoken_form(t, want):
    assert sp.josa_display(t) == want


def test_prepare_caption_and_speech_agree_on_josa():
    p = sp.prepare({"narr": "x", "say": "예측 오차 0.2846를 더해 0.4531입니다. 최대 차이는 0로, 같습니다."})
    assert p["captions"][0] == "예측 오차 0.2846을 더해 0.4531입니다." and p["speech"][0].startswith("예측 오차 영 점 이팔사육을")
    assert p["captions"][1].startswith("최대 차이는 0으로,") and p["speech"][1].startswith("최대 차이는 영으로,")


def test_run_lab_float_tolerance():
    from tools.codewalk.common import run_lab
    exp = "a = 0.12345678\nb = [1.0000004, 2.0]"
    near = run_lab("print('a = 0.12345679')\nprint('b = [1.0000005, 2.0]')", exp, pure=True, rtol=2e-5)
    assert near["ok"] and near["tol"]
    strict = run_lab("print('a = 0.12345679')\nprint('b = [1.0000005, 2.0]')", exp, pure=True)
    assert not strict["ok"]
    far = run_lab("print('a = 0.1240')\nprint('b = [1.0000005, 2.0]')", exp, pure=True, rtol=2e-5)
    assert not far["ok"]
    text = run_lab("print('c = 0.12345678')\nprint('b = [1.0000004, 2.0]')", exp, pure=True, rtol=2e-5)
    assert not text["ok"]                                               # 숫자 밖 글자가 다르면 허용오차와 무관하게 실패


def test_check_say_number_provenance():
    from tools.codewalk.check_say import matches, nums
    assert nums("값 0.001493은, 2020-03의 -0.023, 1,234개") == ["0.001493", "2020", "03", "-0.023", "1234"]
    pool = [0.0014932, -0.02259, 3.626e-4]
    assert matches("0.001493", pool) and matches("-0.023", pool) and matches("3.626e-04", pool)
    assert not matches("0.0015", [0.0014]) and not matches("6", pool)


@pytest.mark.parametrize("t, want", [
    ("변수(빈 칸은 x̂ 으로 채운 값)로 보면", "변수(빈 칸은 x̂ 으로 채운 값)로 보면"),   # 괄호 뒤 조사는 괄호 앞 낱말(변수)에 맞춤
    ("테스트 창(이 폴드 59개)을 3개씩", "테스트 창(이 폴드 59개)을 3개씩"),
    ("f(잊기 문)가 클수록", "f(잊기 문)가 클수록"),                                  # f → 에프
    ("경로 출력은", "경로 출력은"),                                                  # 한글 낱말 끝(경로)은 손대지 않음
    ("사이 92개", "사이 92개"),
    ("있는데, 이 가중치에서는", "있는데, 이 가중치에서는"),                          # 쉼표 뒤 '이' 는 지시어
    ("x̂ 로 채웁니다", "x̂ 으로 채웁니다"),                                          # 띄어 쓴 조사도 소리에 맞춤
    ("x̂·ẑ 를 반씩", "x̂·ẑ 을 반씩"),                                                # 연산 기호로 이은 말은 마지막 항
    ("h [3, 128] 를 넣고", "h [3, 128] 을 넣고"),                                  # 모양 목록은 마지막 항(백이십팔)
    ("[3, 11, 34] 를 만듭니다", "[3, 11, 34] 를 만듭니다"),
    ("한 epoch 은 학습", "한 epoch 은 학습"),                                        # 여러 글자 영어 낱말은 읽는 법이 사람마다 달라 그대로
    ("가중치를 1/6(= 평균)로 시작", "가중치를 1/6(= 평균)로 시작"),                   # 분수는 그대로
    ("ε 는 표준정규", "ε 은 표준정규"),
])
def test_josa_display_rules(t, want):
    assert sp.josa_display(t) == want


def test_months_and_counters():
    assert sp.speakify("2018-10부터 2020-06까지") == "이천십팔 년 시월부터 이천이십 년 유월까지"
    assert sp.speakify("18줄을 6번, 24칸 가운데 0칸, 4자리") == "열여덟 줄을 여섯 번, 스물네 칸 가운데 영 칸, 네 자리"
    assert sp.speakify("값이 줄었습니다") == "값이 줄었습니다"
    assert sp.speakify("비교가 6줄이고, 10줄입니다") == "비교가 여섯 줄이고, 열 줄입니다"   # '줄이고'(동사 줄이다)로 보던 오류(11-09)
    assert sp.speakify("값을 3 줄여서") == "값을 삼 줄여서" and sp.speakify("2 줄인 값") == "이 줄인 값"


def test_run_lab_tutor_flag():
    from tools.codewalk.common import run_lab
    assert run_lab("print(1)", "1", pure=True)["tutor"] is True
    assert run_lab("print(1)", "1", pure=True, tutor=False)["tutor"] is False
    assert run_lab("print(1)", "1", pure=False, tutor=True)["tutor"] is False          # numpy 코드는 Python Tutor 로 열지 않음
