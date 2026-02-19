# FinLearn AI Backend

FastAPI backend that connects the React frontend to:
- RAG Chat (Gemini + Knowledge Base)
- Chart Analysis (CV Models)
- Stock Screener (Polygon API)

## Setup

### 1. Create virtual environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create `.env` file

```bash
cat > .env << 'EOF'
GEMINI_API_KEY=your_gemini_api_key
POLYGON_API_KEY=your_polygon_api_key
EOF
```

### 3. Run the server

```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/chat` | POST | RAG chat with knowledge base |
| `/api/analyze-chart` | POST | Analyze chart image (S/R + Trend) |
| `/api/stocks` | POST | Get stock data with AI signals |

## Example Requests

### Chat
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is dollar cost averaging?"}'
```

### Analyze Chart
```bash
curl -X POST http://localhost:8000/api/analyze-chart \
  -F "file=@chart.png"
```

### Get Stocks
```bash
curl -X POST http://localhost:8000/api/stocks \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL"]}'
```

## Architecture

```
React Frontend (port 5173)
    ↓
FastAPI Backend (port 8000)
    ├── /api/chat → RAG (quantcademy-app/rag/)
    ├── /api/analyze-chart → CV Models (chart-vision/)
    └── /api/stocks → Polygon API
```
