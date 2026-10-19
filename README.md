# Can an AI Agent Tell When It's Being Told the Truth?
Streamlit app for the CA3 study on confidence bias in agentic AI decision-making.
Authors: Kushbu Agrawal and Sonalika Singha.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (see commands below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create app -> pick the repo, branch `main`, main file `app.py`.
4. (Advanced settings) choose Python 3.11 or 3.12.
5. Deploy. Optional: add `ANTHROPIC_API_KEY = "..."` under Settings -> Secrets for the LLM tab.

## Push to GitHub
```bash
git init
git add .
git commit -m "Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Notes
- Defaults (3 seeds, 100 facts/domain) keep runs quick on the free tier. Use 10 seeds and 300 facts/domain to match the report.
- Everything is seeded, so the same settings always give the same numbers.
- Optional notebook extensions (MarianMT, DistilBART, LIAR, spaCy) are left out to keep the app light.
