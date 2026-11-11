# 화면 (React + TypeScript + Vite)

빌드된 결과(`dist/`)가 저장소에 포함되어 있어 Node 없이도 `run_server.bat` 만으로 화면이 뜹니다(FastAPI 가 dist 를 서빙).

화면 코드를 고칠 때만 Node 가 필요합니다:
```
cd web
npm install          # 처음 한 번
npm run dev          # 개발 서버 http://localhost:5173 (API 는 8000 으로 프록시)
npm run build        # dist/ 갱신 → run_server.bat 로 확인
```
구조: `src/App.tsx`(메뉴) · `ExperimentForm.tsx`(새 실험) · `Jobs.tsx`(작업·로그) · `Results.tsx`(리더보드·상세·에쿼티 차트) · `Variables.tsx`(변수 사전, 읽기 전용) · `api.ts`(API 호출)
규칙: 화면은 계산하지 않는다. 지표·판정 기준의 수치는 모두 API(엔진)에서 온 값을 그대로 보여준다.
