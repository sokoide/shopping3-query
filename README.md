# How to run

```bash
export GRAFANA_API_KEY="hogehoge..."
export GRAFANA_URL="http://lab3:13000"
streamlit run main.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection=false --server.headless true
```
