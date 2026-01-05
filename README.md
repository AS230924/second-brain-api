# Second Brain - Railway Deployment

## Quick Deploy

### Option 1: Deploy via Railway Dashboard

1. Push this folder to a GitHub repo
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Add environment variables (see below)
6. Deploy!

### Option 2: Deploy via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

## Environment Variables

Set these in Railway dashboard → Variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Yes | Your Anthropic API key for Claude |
| `EMBEDDING_PROVIDER` | No | Default: `sentence-transformers` |
| `EMBEDDING_MODEL` | No | Default: `all-MiniLM-L6-v2` |
| `PORT` | Auto | Set automatically by Railway |

## After Deployment

1. Get your Railway URL (e.g., `https://second-brain-xxx.up.railway.app`)
2. Test the health endpoint: `curl https://your-url.railway.app/health`
3. Use this URL as `VITE_API_URL` in your Lovable frontend

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Query knowledge base |
| POST | `/connect` | Find connections |
| POST | `/ingest/file` | Upload file |
| POST | `/ingest` | Ingest text content |
| GET | `/sources` | List all sources |
| DELETE | `/sources/{id}` | Delete a source |
| GET | `/stats` | Get statistics |

## File Structure

```
railway-deploy/
├── api.py              # FastAPI server (entry point)
├── main.py             # Core Second Brain logic
├── Procfile            # Process definition
├── requirements.txt    # Python dependencies
├── railway.json        # Railway config
├── config/
│   └── config.example.yaml
├── data/               # Runtime data (auto-created)
└── src/
    ├── models.py
    ├── agent/
    ├── ingestion/
    ├── processing/
    ├── storage/
    └── evaluation/
```

## Troubleshooting

### First request is slow
The embedding model loads on first request (~30s). Subsequent requests are fast.

### Out of memory
Railway free tier has 512MB RAM. The embedding model needs ~400MB.
- Upgrade to paid plan, OR
- Use OpenAI embeddings instead (set `EMBEDDING_PROVIDER=openai`)

### CORS errors
The API allows these origins by default:
- `localhost:3000`, `localhost:5173`
- `*.lovable.app`, `*.lovableproject.com`

Add your domain in `api.py` if needed.
