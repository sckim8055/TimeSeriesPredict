"""원본 BIVA 소스(D:/test/SVU_SCT_FIN-main/Model/BIVA/model) 의 사본.

- 원본 파일을 그대로 복사한 뒤, 플랫폼에서 돌기 위해 꼭 필요한 곳만 고쳤다. 고친 곳은 모두 `[TSP 수정 #번호]` 주석이 있고
  바로 위에 `# 원본:` 으로 원래 줄을 남겼다. 목록은 PATCHES.md, 원본과의 차이는 `python tools/biva_src_diff.py`.
- 신경망 구조·손실·학습 규칙은 바꾸지 않았다. 학습 루프(exp_main.train)는 models/biva.py 가 원본 순서 그대로 재현한다.
"""
