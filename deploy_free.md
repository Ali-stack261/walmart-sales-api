# Deploying walmart-sales-api + UI — Free Tier Guide

Architecture:
```
[ Streamlit UI ]  --HTTP-->  [ FastAPI on Render ]
Streamlit Community Cloud       (Docker, free web service)
```

Two separate free services, each on the platform built for it. Total cost: $0.

---

## Step 0 — Add the Streamlit app to your repo

1. Save `streamlit_app.py` (provided) into the **root** of your repo, next to `readme.md` and `Dockerfile`.
2. Add these two lines to `requirements.txt` if not already present:
   ```
   streamlit>=1.35.0
   requests>=2.31.0
   ```
3. Commit and push:
   ```bash
   git add streamlit_app.py requirements.txt
   git commit -m "Add Streamlit UI"
   git push
   ```

---

## Step 1 — Deploy the API on Render (free)

Render has a free tier for web services and reads your existing `Dockerfile` directly — no changes needed to your API code.

1. Go to https://render.com and sign up / log in with GitHub.
2. Click **New +** → **Web Service**.
3. Connect your `walmart-sales-api` GitHub repo.
4. Configure:
   - **Environment:** Docker (Render auto-detects your `Dockerfile`)
   - **Instance Type:** Free
   - **Branch:** `main`
5. Click **Create Web Service**. Render will build the image and deploy it.
6. Once live, you'll get a URL like:
   ```
   https://walmart-sales-api-xxxx.onrender.com
   ```
7. Test it:
   ```bash
   curl https://walmart-sales-api-xxxx.onrender.com/health
   ```

### ⚠️ Free tier behavior to know about
- Render's free web services **spin down after ~15 minutes of inactivity** and take 30-60 seconds to wake up on the next request. This is normal — the Streamlit app above already shows a friendly message for this case.
- Free tier has limited monthly hours; fine for a portfolio/demo project, not for production traffic.
- Your model needs to actually exist in the image at build time. Since your CI now runs the full pipeline (ingest → preprocess → features → train) before tests pass, make sure `models/best_model.joblib` is either:
  - produced by a `RUN` step in the `Dockerfile` (e.g. `RUN python -m src.ingest && python -m src.preprocessing && python -m src.feature_engineering && python -m src.train`), **or**
  - already committed to the repo (Option B from the earlier CI fix).

  Right now your `Dockerfile` just does `COPY . .` — if `models/best_model.joblib` isn't tracked in git, Render's build **won't have a model to serve** and `/predict` will fail at container startup (same root cause as the earlier CI bug, just in a different environment). Pick one of the two options above before deploying.

---

## Step 2 — Deploy the UI on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Select your `walmart-sales-api` repo, branch `main`, and set:
   - **Main file path:** `streamlit_app.py`
4. Before deploying, add a secret so the UI knows where your API lives:
   - In the app's **Settings → Secrets**, add:
     ```toml
     API_URL = "https://walmart-sales-api-xxxx.onrender.com"
     ```
5. Click **Deploy**. You'll get a public URL like:
   ```
   https://your-app-name.streamlit.app
   ```

That's it — the UI is now live and talking to your API.

---

## Step 3 — Verify end to end

1. Open your Streamlit URL.
2. Check the sidebar shows **"API is online"** (if it shows an error and you just deployed, wait ~60s for Render's free instance to wake up, then refresh).
3. Go to the **Single Prediction** tab, fill in values, click **Predict** — you should see a predicted sales figure.
4. Try the **Batch Prediction** tab with the downloadable CSV template.
5. Try the **Drift Report** tab.

---

## Alternative: one free host instead of two

If you'd rather run everything as a single deployment (simpler ops, no cross-service networking), **Hugging Face Spaces** supports Docker and can host both the API and a Streamlit UI in one container using a small startup script that launches both processes. This trades simplicity of setup (one platform) for a bit more Docker complexity (running two processes in one container). Worth considering later, but the two-service Render + Streamlit Cloud setup above is the more standard path and easier to debug independently — recommended for now.

---

## Costs recap

| Service | Free tier limit | Good enough for this project? |
|---|---|---|
| Render (API) | Free web service, spins down after inactivity, limited monthly hours | ✅ Yes, for a portfolio demo |
| Streamlit Community Cloud (UI) | Free, public apps, some resource limits | ✅ Yes |

No credit card required for either at the free tier.
