from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent
CSV_PATH = DATA_DIR / "E-commerce Dataset.csv"

app = FastAPI(title="AI Chatbot & Data Query Interface")

# Keep a single shared connection for the in-memory dataset.
_DB: sqlite3.Connection | None = None
_TABLE_NAME = "ecommerce"
_COLUMNS: list[str] = []


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    hints: list[str]
    suggested_sql: str | None = None


class QueryRequest(BaseModel):
    sql: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int


class DatasetInfo(BaseModel):
    table: str
    columns: list[str]
    sample_queries: list[str]


def _load_dataset(csv_path: Path) -> sqlite3.Connection:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    global _COLUMNS
    _COLUMNS = list(df.columns)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(_TABLE_NAME, conn, if_exists="replace", index=False)
    return conn


def _find_column(target: str) -> str | None:
    target_lower = target.lower()
    for column in _COLUMNS:
        if column.lower() == target_lower:
            return column
    return None


def _suggest_sql(message: str) -> str | None:
    text = message.strip().lower()
    if not text:
        return None

    total_rows = "SELECT COUNT(*) AS total_rows FROM ecommerce"
    if any(keyword in text for keyword in ["total rows", "count rows", "how many", "total records", "total orders"]):
        return total_rows

    group_targets = {
        "product_category": ["category", "product category"],
        "product": ["product"],
        "device_type": ["device"],
        "gender": ["gender"],
        "payment_method": ["payment", "payment method"],
        "order_priority": ["priority"],
        "order_date": ["date", "order date"],
    }

    def pick_group_column() -> str | None:
        for column_key, keywords in group_targets.items():
            if any(keyword in text for keyword in keywords):
                return _find_column(column_key)
        return None

    group_column = pick_group_column()
    if group_column:
        if any(keyword in text for keyword in ["top", "most", "highest", "breakdown", "by"]):
            return (
                f"SELECT {group_column}, COUNT(*) AS count "
                f"FROM ecommerce GROUP BY {group_column} ORDER BY count DESC LIMIT 10"
            )

    if any(keyword in text for keyword in ["sales", "revenue"]):
        sales_column = _find_column("sales")
        if sales_column is None:
            return None
        if group_column:
            return (
                f"SELECT {group_column}, SUM({sales_column}) AS total_sales "
                f"FROM ecommerce GROUP BY {group_column} ORDER BY total_sales DESC LIMIT 10"
            )
        return f"SELECT SUM({sales_column}) AS total_sales FROM ecommerce"

    if "profit" in text:
        profit_column = _find_column("profit")
        if profit_column is None:
            return None
        if group_column:
            return (
                f"SELECT {group_column}, SUM({profit_column}) AS total_profit "
                f"FROM ecommerce GROUP BY {group_column} ORDER BY total_profit DESC LIMIT 10"
            )
        return f"SELECT SUM({profit_column}) AS total_profit FROM ecommerce"

    if "discount" in text:
        discount_column = _find_column("discount")
        if discount_column is None:
            return None
        if group_column:
            return (
                f"SELECT {group_column}, AVG({discount_column}) AS avg_discount "
                f"FROM ecommerce GROUP BY {group_column} ORDER BY avg_discount DESC LIMIT 10"
            )
        return f"SELECT AVG({discount_column}) AS avg_discount FROM ecommerce"

    if "quantity" in text:
        quantity_column = _find_column("quantity")
        if quantity_column is None:
            return None
        if group_column:
            return (
                f"SELECT {group_column}, SUM({quantity_column}) AS total_quantity "
                f"FROM ecommerce GROUP BY {group_column} ORDER BY total_quantity DESC LIMIT 10"
            )
        return f"SELECT SUM({quantity_column}) AS total_quantity FROM ecommerce"

    if group_column:
        return (
            f"SELECT {group_column}, COUNT(*) AS count "
            f"FROM ecommerce GROUP BY {group_column} ORDER BY count DESC LIMIT 10"
        )

    return total_rows


def _is_safe_select(sql: str) -> bool:
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        return False
    if ";" in normalized:
        return False
    blocked = ["pragma", "attach", "detach", "drop", "alter", "insert", "update", "delete"]
    return not any(keyword in normalized for keyword in blocked)


@app.on_event("startup")
def _startup() -> None:
    global _DB
    _DB = _load_dataset(CSV_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
        return """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AI Chatbot & Data Query Interface</title>
        <style>
            @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap");

            :root {
                --ink: #0f1b1b;
                --muted: #4a5a5a;
                --accent: #e76f51;
                --accent-2: #2a9d8f;
                --paper: #f5f0ea;
                --panel: #ffffff;
                --shadow: rgba(15, 27, 27, 0.12);
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: "Space Grotesk", system-ui, -apple-system, sans-serif;
                color: var(--ink);
                background: var(--paper);
                min-height: 100vh;
            }

            header {
                padding: 32px 24px 8px;
                max-width: 1200px;
                margin: 0 auto;
            }

            header h1 {
                margin: 0 0 6px;
                font-size: 32px;
                letter-spacing: -0.02em;
            }

            header p {
                margin: 0;
                color: var(--muted);
            }

            main {
                display: grid;
                grid-template-columns: minmax(240px, 1fr) minmax(320px, 2fr);
                gap: 20px;
                padding: 24px;
                max-width: 1200px;
                margin: 0 auto 40px;
            }

            .card {
                background: var(--panel);
                border-radius: 18px;
                box-shadow: 0 16px 36px var(--shadow);
                padding: 20px;
                animation: rise 500ms ease;
            }

            @keyframes rise {
                from {
                    transform: translateY(12px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }


            textarea,
            input {
                width: 100%;
                border-radius: 12px;
                border: 1px solid #e2ddd6;
                padding: 12px;
                font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
                background: #fbfaf7;
            }

            textarea {
                min-height: 120px;
                resize: vertical;
            }

            button {
                appearance: none;
                border: none;
                border-radius: 12px;
                padding: 10px 16px;
                font-weight: 600;
                cursor: pointer;
                background: var(--accent);
                color: #fff;
                transition: transform 150ms ease, box-shadow 150ms ease;
                box-shadow: 0 6px 14px rgba(15, 27, 27, 0.18);
            }

            button.secondary {
                background: var(--accent-2);
                box-shadow: 0 8px 18px rgba(42, 157, 143, 0.3);
            }

            button:hover {
                transform: translateY(-2px);
            }

            .stack {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .row {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .muted {
                color: var(--muted);
                font-size: 14px;
            }

            .queries {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            .chip {
                background: #e1d7cd;
                border: 1px dashed #d5cec6;
                padding: 6px 10px;
                border-radius: 10px;
                font-size: 12px;
                cursor: pointer;
            }

            .results {
                overflow-x: auto;
                border-radius: 12px;
                border: 1px solid #ebe5dd;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }

            th,
            td {
                padding: 10px 12px;
                border-bottom: 1px solid #efe9e2;
                text-align: left;
            }

            th {
                background: #fbf7f0;
                font-weight: 600;
            }

            @media (max-width: 900px) {
                main {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>AI Chatbot & Data Query Interface</h1>
            <p>Ask quick questions or run SQL against the e-commerce dataset.</p>
        </header>
        <main>
            <section class="card stack">
                <div class="stack">
                    <div class="row" style="justify-content: space-between;">
                        <strong>Dataset</strong>
                        <span id="dataset-status" class="muted">Loading...</span>
                    </div>
                    <div class="muted">Columns</div>
                    <div id="columns" class="muted"></div>
                </div>
                <div class="stack">
                    <div class="muted">Sample queries</div>
                    <div id="samples" class="queries"></div>
                </div>
                <div class="stack">
                    <div class="row">
                        <input id="chat-input" type="text" placeholder="Ask a question" />
                        <button id="chat-btn" class="secondary">Chat</button>
                    </div>
                    <div id="chat-reply" class="muted">Tip: Ask for totals, breakdowns, or top values.</div>
                </div>
            </section>
            <section class="card stack">
                <div class="row" style="justify-content: space-between;">
                    <strong>SQL Query</strong>
                    <span id="query-status" class="muted">Ready</span>
                </div>
                <textarea id="sql-input" placeholder="SELECT * FROM ecommerce LIMIT 10"></textarea>
                <div class="row" style="justify-content: flex-end;">
                    <button id="run-btn">Run query</button>
                </div>
                <div class="results" id="results">
                    <table>
                        <thead id="results-head"></thead>
                        <tbody id="results-body"></tbody>
                    </table>
                </div>
                <div id="row-count" class="muted"></div>
            </section>
        </main>
        <script>
            const datasetStatus = document.getElementById("dataset-status");
            const columnsEl = document.getElementById("columns");
            const samplesEl = document.getElementById("samples");
            const sqlInput = document.getElementById("sql-input");
            const runBtn = document.getElementById("run-btn");
            const queryStatus = document.getElementById("query-status");
            const resultsHead = document.getElementById("results-head");
            const resultsBody = document.getElementById("results-body");
            const rowCount = document.getElementById("row-count");
            const chatBtn = document.getElementById("chat-btn");
            const chatInput = document.getElementById("chat-input");
            const chatReply = document.getElementById("chat-reply");

            async function loadDataset() {
                try {
                    const res = await fetch("/dataset");
                    const data = await res.json();
                    datasetStatus.textContent = "Ready";
                    columnsEl.textContent = data.columns.join(", ");
                    samplesEl.innerHTML = "";
                    data.sample_queries.forEach((query) => {
                        const chip = document.createElement("button");
                        chip.className = "chip";
                        chip.textContent = query;
                        chip.addEventListener("click", () => {
                            sqlInput.value = query;
                        });
                        samplesEl.appendChild(chip);
                    });
                } catch (error) {
                    datasetStatus.textContent = "Error";
                }
            }

            function renderResults(columns, rows) {
                resultsHead.innerHTML = "";
                resultsBody.innerHTML = "";
                if (!columns || columns.length === 0) {
                    return;
                }
                const headRow = document.createElement("tr");
                columns.forEach((col) => {
                    const th = document.createElement("th");
                    th.textContent = col;
                    headRow.appendChild(th);
                });
                resultsHead.appendChild(headRow);
                rows.forEach((row) => {
                    const tr = document.createElement("tr");
                    row.forEach((cell) => {
                        const td = document.createElement("td");
                        td.textContent = cell === null ? "" : String(cell);
                        tr.appendChild(td);
                    });
                    resultsBody.appendChild(tr);
                });
            }

            runBtn.addEventListener("click", async () => {
                queryStatus.textContent = "Running...";
                rowCount.textContent = "";
                try {
                    const res = await fetch("/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ sql: sqlInput.value }),
                    });
                    const data = await res.json();
                    if (!res.ok) {
                        throw new Error(data.detail || "Query failed");
                    }
                    renderResults(data.columns, data.rows);
                    rowCount.textContent = `Rows: ${data.row_count}`;
                    queryStatus.textContent = "Done";
                } catch (error) {
                    queryStatus.textContent = "Error";
                    rowCount.textContent = error.message;
                }
            });

            chatBtn.addEventListener("click", async () => {
                chatReply.textContent = "Thinking...";
                try {
                    const res = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ message: chatInput.value }),
                    });
                    const data = await res.json();
                    if (!res.ok) {
                        throw new Error(data.detail || "Chat failed");
                    }
                    const hints = data.hints.length ? ` Hints: ${data.hints.join(" | ")}` : "";
                    chatReply.textContent = `${data.reply}${hints}`;
                    if (data.suggested_sql) {
                        sqlInput.value = data.suggested_sql;
                    }
                } catch (error) {
                    chatReply.textContent = error.message;
                }
            });

            loadDataset();
        </script>
    </body>
</html>
"""


@app.get("/dataset", response_model=DatasetInfo)
def dataset_info() -> DatasetInfo:
    if _DB is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    cursor = _DB.execute(f"PRAGMA table_info({_TABLE_NAME});")
    columns = [row[1] for row in cursor.fetchall()]

    sample_queries = [
        f"SELECT * FROM {_TABLE_NAME} LIMIT 5",
        f"SELECT COUNT(*) AS total_rows FROM {_TABLE_NAME}",
    ]
    if columns:
        sample_queries.append(
            f"SELECT {columns[0]} , COUNT(*) AS count FROM {_TABLE_NAME} GROUP BY {columns[0]} ORDER BY count DESC LIMIT 10"
        )

    return DatasetInfo(table=_TABLE_NAME, columns=columns, sample_queries=sample_queries)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    suggested_sql = _suggest_sql(payload.message)
    if suggested_sql:
        reply = "I put a SQL suggestion into the editor."
        hints = [
            "Click Run query to execute it",
            "You can edit the SQL before running",
        ]
        return ChatResponse(reply=reply, hints=hints, suggested_sql=suggested_sql)

    reply = "I could not map that to a query yet. Try asking for totals or breakdowns."
    hints = [
        "Example: total sales by product category",
        "Example: top 10 device types",
    ]
    return ChatResponse(reply=reply, hints=hints)


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if _DB is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    sql = re.sub(r"\s+", " ", payload.sql.strip())
    if not _is_safe_select(sql):
        raise HTTPException(status_code=400, detail="Only single SELECT statements are allowed")

    try:
        cursor = _DB.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(columns=columns, rows=[list(row) for row in rows], row_count=len(rows))
