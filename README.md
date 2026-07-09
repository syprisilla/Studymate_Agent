# StudyMate Ready

## 실행
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# .env에 OPENAI_API_KEY 입력
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

브라우저: http://127.0.0.1:8000

PDF를 올리면 서버의 `data/pdfs`에 저장되고, 같은 세션에서 바로 질문/요약/퀴즈가 됩니다.
OPENAI_API_KEY가 없으면 PDF 텍스트 일부를 기반으로 한 fallback 응답만 동작합니다.
