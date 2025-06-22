import json
from datetime import datetime, timedelta, timezone
import streamlit as st
import requests
import os
from openai import AzureOpenAI
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

logging.basicConfig(level=logging.INFO)

key = os.getenv("KEY")
model = "gpt-4.1"

client = AzureOpenAI(
    api_key=key,
    api_version="2024-12-01-preview",
    azure_endpoint="https://sokoide-openai.openai.azure.com/"
)

# MCP Grafana server configuration
MCP_SERVER_PATH = "mcp-grafana"
# MCP_SERVER_ARGS = ["-debug"]
MCP_SERVER_ARGS = [""]


server_params = StdioServerParameters(
    command=MCP_SERVER_PATH,
    args=MCP_SERVER_ARGS,
    env=os.environ,
)

async def test_mcp_connection():
    """Test MCP connection and list available tools"""
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [tool.name for tool in tools.tools]
    except Exception as e:
        logging.error(f"MCP connection test failed: {e}")
        return []


async def query_logs_with_mcp(query_text):
    """Query Loki logs using Grafana MCP"""
    # Generate LogQL from natural language using AI first
    prompt_logql = f"""
You are an expert in querying Loki logs. Given the following user question:

"{query_text}"

Return only a single line with a suitable LogQL query to find relevant log lines from a job named sokoide-shopping.
The logs are JSON and include 'body' fields with order information.

Make sure that regex in LogQL requires ".*" to match before or after the regex.

If product(s) are purchased, the log is something like this:
    "{{"body":"Order placed. User:Scott, ID:ORD-1749377312422-8J7B1, Timestamp:2025-06-08T10:08:32.422Z, Items:17.96, Organic Apples (ID: 1), Qty: 2, Price: $3.99,Free-Range Eggs (ID: 3), Qty: 2, Price: $4.99","severity":"INFO","resources":{{"service.name":"sokoide-shopping"}},"instrumentation_scope":{{"name":"sokoide-logger"}}}}"

If there is an error/failure, log is something like this:
    "{{"body":"Checkout failed: User:scott, ID:ORD-1749367435824-4G4GL, Reason: Credit card service is down. Please try again later.","severity":"ERROR","resources":{{"service.name":"sokoide-shopping"}},"instrumentation_scope":{{"name":"sokoide-logger"}}}}"

Respond with only the LogQL query (no comments or explanation).
"""
    logql_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_logql}],
        temperature=0
    )
    logql_query = logql_response.choices[0].message.content.strip()
    logging.info(f"Generated LogQL: {logql_query}")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the client session
                await session.initialize()

                # List available tools for debugging
                tools = await session.list_tools()
                available_tools = [tool.name for tool in tools.tools]
                logging.info(f"Available MCP tools: {available_tools}")

                # Get detailed info about all Loki-related tools
                loki_tools = []
                for tool in tools.tools:
                    if "loki" in tool.name.lower():
                        loki_tools.append(tool.name)
                        logging.info(f"{tool.name} tool details: {tool}")
                        if hasattr(tool, 'inputSchema'):
                            logging.info(f"{tool.name} input schema: {tool.inputSchema}")
                        if hasattr(tool, 'description'):
                            logging.info(f"{tool.name} description: {tool.description}")

                logging.info(f"All Loki-related tools found: {loki_tools}")

                if "query_loki_logs" not in available_tools:
                    raise ValueError(f"query_loki_logs tool not available. Available tools: {available_tools}")

                # First, let's try to list datasources to understand what's available
                loki_datasource_uid = None
                loki_datasource_name = None
                try:
                    if "list_datasources" in available_tools:
                        datasources = await session.call_tool("list_datasources", {})
                        logging.info(
                            f"Full datasources response: {datasources}")

                        # More detailed parsing of datasources
                        if hasattr(datasources, 'content'):
                            logging.info(f"Datasources content type: {type(datasources.content)}")
                            logging.info(f"Datasources content: {datasources.content}")

                            content = datasources.content
                            if isinstance(content, list):
                                for item in content:
                                    logging.info(f"Datasource item: {item}")

                                    # Extract the JSON text from the TextContent object
                                    if hasattr(item, 'text'):
                                        json_text = item.text
                                        try:
                                            import json
                                            datasources_list = json.loads(
                                                json_text)
                                            logging.info(f"Parsed datasources JSON: {datasources_list}")

                                            # Find the datasource with type "loki"
                                            for ds in datasources_list:
                                                if ds.get('type') == 'loki':
                                                    loki_datasource_uid = ds.get(
                                                        'uid')
                                                    loki_datasource_name = ds.get(
                                                        'name')
                                                    logging.info(
                                                        f"Found Loki datasource - UID: {loki_datasource_uid}, Name: {loki_datasource_name}")
                                                    break

                                            if loki_datasource_uid and loki_datasource_name:
                                                break

                                        except json.JSONDecodeError as e:
                                            logging.warning(
                                                f"Could not parse datasources JSON: {e}")
                                            # Fallback to regex parsing
                                            item_str = str(item)
                                            if 'loki' in item_str.lower():
                                                logging.info(
                                                    f"Found Loki datasource item (fallback): {item}")
                                    else:
                                        # Fallback for non-TextContent items
                                        item_str = str(item)
                                        if 'loki' in item_str.lower():
                                            logging.info(
                                                f"Found Loki datasource item (string): {item}")
                            elif isinstance(content, str):
                                if 'loki' in content.lower():
                                    logging.info(
                                        f"Found Loki in datasources string: {content}")
                except Exception as ds_error:
                    logging.warning(f"Could not list datasources: {ds_error}")

                # Use MCP to query Loki logs
                # tokyo_tz = timezone('Asia/Tokyo')
                # end = datetime.now(tokyo_tz)
                end = datetime.now(timezone.utc)
                start = end - timedelta(hours=12)

                logging.info(f"* Querying logs from {start.isoformat()} to {end.isoformat()}")

                # Build list of datasource identifiers to try
                # Since UID seems to have a bug, prioritize names
                datasource_identifiers_to_try = []

                # Try the actual name first (since UID has a bug)
                if loki_datasource_name:
                    datasource_identifiers_to_try.append(loki_datasource_name)

                # Add common names as fallback
                datasource_identifiers_to_try.extend(
                    ["loki", "Loki", "loki-datasource"])

                # Try UID last (since it seems to have a bug)
                if loki_datasource_uid:
                    datasource_identifiers_to_try.append(loki_datasource_uid)

                base_params = {
                    "logql": logql_query,
                    "startRfc3339": start.isoformat(),
                    "endRfc3339": end.isoformat(),
                    "limit": 100
                }

                # Try each datasource identifier (names first, since UID has a bug)
                for ds_identifier in datasource_identifiers_to_try:
                    if loki_datasource_uid:
                        params = {
                            **base_params,
                            "datasourceUid": loki_datasource_uid
                        }
                        logging.info(f"Trying datasourceUid: {loki_datasource_uid} with params: {params}")
                        try:
                            result = await session.call_tool("query_loki_logs", params)
                            logging.info(f"SUCCESS with datasourceUid: {loki_datasource_uid}")
                            logging.info(f"MCP result type: {type(result)}")
                            logging.info(f"MCP result: {result}")
                            return logql_query, result.content
                        except Exception as uid_error:
                            logging.error(f"Failed with datasourceUid '{loki_datasource_uid}': {uid_error}")
                    else:
                        logging.warning(
                            "Loki datasource UID not found, cannot query Loki logs using UID.")

                # If all datasource names failed, try without datasource parameter
                logging.info(
                    "All datasource names failed, trying without datasource parameter...")
                params = base_params
                logging.info(f"Calling query_loki_logs with params: {params}")
                result = await session.call_tool("query_loki_logs", params)

                logging.info(f"MCP result type: {type(result)}")
                logging.info(f"MCP result: {result}")

                return logql_query, result.content

    except ExceptionGroup as eg:
        logging.error(f"MCP ExceptionGroup with {len(eg.exceptions)} exceptions:")
        for i, exc in enumerate(eg.exceptions):
            logging.error(f"  Exception {i+1}: {type(exc).__name__}: {exc}")
            # If it's another ExceptionGroup, unwrap it further
            if isinstance(exc, ExceptionGroup):
                for j, inner_exc in enumerate(exc.exceptions):
                    logging.error(f"    Inner Exception {j+1}: {type(inner_exc).__name__}: {inner_exc}")
        # Return the query and empty result so the app doesn't crash
        return logql_query, []
    except Exception as e:
        logging.error(f"MCP Error: {e}")
        logging.error(f"Error type: {type(e)}")
        # Return the query and empty result so the app doesn't crash
        return logql_query, []


def run_async_query(query_text):
    """Wrapper to run async MCP query in Streamlit"""
    return asyncio.run(query_logs_with_mcp(query_text))


st.title("🛒 Sokoide Shopping Log Explorer")

# Add a test connection button
if st.button("Test MCP Connection"):
    try:
        available_tools = asyncio.run(test_mcp_connection())
        if available_tools:
            st.success(f"✅ MCP Connection successful! Available tools: {', '.join(available_tools)}")
        else:
            st.error("❌ MCP Connection failed or no tools available")
    except Exception as e:
        st.error(f"❌ MCP Connection error: {e}")

question = st.text_input("What can I help?", "What did Scott buy?")

if st.button("Execute:"):
    try:
        # Query logs using MCP
        logql_query, mcp_result = run_async_query(question)

        logging.info(f"LogQL: {logql_query}")
        st.code(logql_query, language="text")

        # Parse MCP result to extract log lines
        log_lines = []
        if isinstance(mcp_result, list):
            for item in mcp_result:
                if isinstance(item, dict):
                    # Extract text content from MCP result
                    if 'text' in item:
                        log_lines.append(item['text'])
                    elif 'content' in item:
                        log_lines.append(item['content'])
                    else:
                        log_lines.append(str(item))
                else:
                    log_lines.append(str(item))
        else:
            # If result is a single text block, try to parse it
            result_str = str(mcp_result)
            if result_str:
                log_lines.append(result_str)

        st.subheader("Queried log")
        for line in log_lines:
            st.text(line)

        # Generate summary using OpenAI
        if log_lines:
            summary_prompt = f"""
Based on the following logs, answer the following question in natural English:
{question}

The log lines are:

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
        else:
            st.warning("No logs found for the query.")

    except Exception as e:
        st.error(f"Error querying logs: {str(e)}")
        logging.error(f"Error: {e}")
