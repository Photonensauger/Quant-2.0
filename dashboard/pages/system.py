"""Page 6: System Health — Once UI style."""

import platform
import sys
from datetime import datetime

import dash
from dash import html, dcc, callback, Output, Input

from dashboard import config

dash.register_page(__name__, path="/system", name="System", order=5)

C = config.COLORS


def _format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _freshness_cls(modified):
    age = datetime.now() - modified
    if age.days < 1:
        return "profit"
    elif age.days < 7:
        return "warning"
    return "loss"


layout = html.Div([
    dcc.Interval(id="system-refresh", interval=30_000, n_intervals=0),
    html.Div(id="system-content"),
])


@callback(
    Output("system-content", "children"),
    Input("system-refresh", "n_intervals"),
)
def update_system(_n):
    # Data freshness
    parquet_files = []
    if config.DATA_DIR.exists():
        for p in sorted(config.DATA_DIR.glob("*.parquet")):
            stat = p.stat()
            parquet_files.append({
                "name": p.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
            })

    data_rows = []
    for f in parquet_files:
        age = datetime.now() - f["modified"]
        age_str = f"{age.days}d {age.seconds // 3600}h ago" if age.days > 0 else f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago"
        cls = _freshness_cls(f["modified"])

        data_rows.append(html.Tr([
            html.Td([
                html.I(className="bi bi-file-earmark-bar-graph",
                       style={"marginRight": "0.5rem", "color": C["accent"]}),
                f["name"],
            ]),
            html.Td(_format_size(f["size_bytes"])),
            html.Td(f["modified"].strftime("%Y-%m-%d %H:%M")),
            html.Td(
                html.Span(age_str, className=f"badge-pill {cls}"),
            ),
        ]))

    data_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Div([
                html.Span("Data Freshness", className="section-title"),
                html.Span(f"{len(parquet_files)} files", className="section-count"),
            ], style={"display": "flex", "flexDirection": "column", "gap": "0.1rem"}),
        ], className="section-header"),
        html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("File"), html.Th("Size"), html.Th("Modified"), html.Th("Age"),
                ])),
                html.Tbody(data_rows if data_rows else [
                    html.Tr(html.Td("No Parquet files found.", colSpan=4,
                                    style={"textAlign": "center", "padding": "2rem"})),
                ]),
            ], className="once-table"),
        ], className="card-static", style={"padding": "0", "overflow": "auto"}),
    ], className="section")

    # Model checkpoints
    model_files = []
    if config.MODELS_DIR.exists():
        for p in sorted(config.MODELS_DIR.glob("*.pt")):
            stat = p.stat()
            model_files.append({
                "name": p.stem,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime),
            })

    model_rows = [
        html.Tr([
            html.Td([
                html.I(className="bi bi-file-earmark-code",
                       style={"marginRight": "0.5rem", "color": C["accent"]}),
                m["name"],
            ]),
            html.Td(f"{m['size_mb']:.2f} MB"),
            html.Td(m["modified"].strftime("%Y-%m-%d %H:%M")),
        ])
        for m in model_files
    ]

    model_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Div([
                html.Span("Model Checkpoints", className="section-title"),
                html.Span(f"{len(model_files)} files", className="section-count"),
            ], style={"display": "flex", "flexDirection": "column", "gap": "0.1rem"}),
        ], className="section-header"),
        html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Model"), html.Th("Size"), html.Th("Modified"),
                ])),
                html.Tbody(model_rows if model_rows else [
                    html.Tr(html.Td("No model checkpoints found.", colSpan=3,
                                    style={"textAlign": "center", "padding": "2rem"})),
                ]),
            ], className="once-table"),
        ], className="card-static", style={"padding": "0", "overflow": "auto"}),
    ], className="section")

    # System info
    import os
    info = [
        ("Python", sys.version.split()[0]),
        ("Platform", platform.platform()),
        ("Architecture", platform.machine()),
        ("CPU Cores", str(os.cpu_count() or "N/A")),
        ("Data Dir", str(config.DATA_DIR)),
        ("Backtest Dir", str(config.BACKTEST_DIR)),
        ("Models Dir", str(config.MODELS_DIR)),
    ]

    try:
        import psutil
        mem = psutil.virtual_memory()
        info.insert(4, ("Memory", f"{mem.available / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB ({mem.percent}%)"))
    except ImportError:
        pass

    sys_rows = [
        html.Tr([
            html.Td(label, style={"color": "var(--text-weak)", "fontFamily": "var(--font-sans)", "fontWeight": "500"}),
            html.Td(value),
        ])
        for label, value in info
    ]

    sys_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("System Information", className="section-title"),
        ], className="section-header"),
        html.Div([
            html.Table([html.Tbody(sys_rows)], className="once-table"),
        ], className="card-static", style={"padding": "0", "overflow": "auto"}),
    ], className="section")

    # Timestamp
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = html.Div(
        html.Span(f"Last refreshed: {now_str} \u2022 auto-refresh every 30s",
                  style={"fontFamily": "var(--font-mono)", "fontSize": "0.62rem", "color": "var(--text-disabled)"}),
        style={"textAlign": "right", "marginTop": "0.5rem"},
    )

    return html.Div([data_section, model_section, sys_section, footer])
