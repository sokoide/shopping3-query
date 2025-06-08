import json
from datetime import datetime, timedelta
import streamlit as st
import requests
import os
from openai import AzureOpenAI
import logging

logging.basicConfig(level=logging.INFO)

key = os.getenv("KEY")
model = "gpt-4.1"

client = AzureOpenAI(
    api_key=key,
    api_version="2024-12-01-preview",
    azure_endpoint="https://sokoide-openai.openai.azure.com/"
)

LOKI_URL = "http://localhost:3100/loki/api/v1/query_range"

st.title("🛒 Sokoide Shopping Log Explorer")
question = st.text_input("What can I help?", "What did Scott buy?")

if st.button("Execute:"):
    prompt_logql = r"""
You are an expert in querying Loki logs. Given the following user question:

"{question}"

Return only a single line with a suitable LogQL query to find relevant log lines from a job named sokoide-shopping.
The logs are JSON and include 'body' fields with order information.

If product(s) are purchased, the log is something like this:
    "{\"body\":\"Order placed. User:Scott, ID:ORD-1749377312422-8J7B1, Timestamp:2025-06-08T10:08:32.422Z, Items:17.96, Organic Apples (ID: 1), Qty: 2, Price: $3.99,Free-Range Eggs (ID: 3), Qty: 2, Price: $4.99\",\"severity\":\"INFO\",\"resources\":{\"service.name\":\"sokoide-shopping\"},\"instrumentation_scope\":{\"name\":\"sokoide-logger\"}}"

If there is an error/failure, log is something like this:
    "{\"body\":\"Checkout failed: User:scott, ID:ORD-1749367435824-4G4GL, Reason: Credit card service is down. Please try again later.\",\"severity\":\"ERROR\",\"resources\":{\"service.name\":\"sokoide-shopping\"},\"instrumentation_scope\":{\"name\":\"sokoide-logger\"}}"

Respond with only the LogQL query (no comments or explanation).
"""
    logql_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_logql}],
        temperature=0
    )
    logging.info(
        f"LogQL: {logql_response.choices[0].message.content}")
    logql_query = logql_response.choices[0].message.content.strip()

    st.code(logql_query, language="text")

    # end = datetime.utcnow()
    end = datetime.now()
    start = end - timedelta(hours=1)
    params = {
        "query": logql_query,
        "start": int(start.timestamp() * 1e9),
        "end": int(end.timestamp() * 1e9),
        "limit": 100,
    }

    r = requests.get(LOKI_URL, params=params)
    data = r.json()
    logging.info(f"Response from Loki: {json.dumps(data, indent=2)}")

    log_lines = []
    for stream in data.get("data", {}).get("result", []):
        for ts, entry in stream.get("values", []):
            try:
                log_entry = json.loads(entry)
                log_lines.append(log_entry["body"])
            except Exception:
                log_lines.append(entry)

    st.subheader("Queried log")
    for line in log_lines:
        st.text(line)

    summary_prompt = f"""
Based on the following logs, answer the following question in natural English:
{question}

The log lines are the following "body" field in JSON format.

{chr(10).join(log_lines)}

Question: {question}
"""
    summary = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3
    )
    answer = summary.choices[0].message.content.strip()

    st.subheader("🧠 Answer:")
    st.write(answer)
