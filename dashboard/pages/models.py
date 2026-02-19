"""Page 4: Model Analytics — Once UI style."""

import json
import os
import subprocess
import sys
import tempfile
import threading

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from dashboard.components.kpi_card import create_kpi_card
from dashboard.data.loader import get_shared_loader
from dashboard import config

dash.register_page(__name__, path="/models", name="Models", order=3)

loader = get_shared_loader()
C = config.COLORS
ASSET_OPTIONS = config.ASSET_OPTIONS
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"

MODEL_REGISTRY = [
    {"name": "DecoderTransformer", "key": "transformer", "icon": "bi bi-diagram-3",
     "desc": "Autoregressive decoder-only transformer for price prediction.",
     "arch": "3 layers, 4 heads, d=64, ff=256", "type": "Transformer"},
    {"name": "ITransformer", "key": "itransformer", "icon": "bi bi-arrows-angle-expand",
     "desc": "Inverted transformer treating features as tokens for multivariate forecasting.",
     "arch": "2 layers, 4 heads, d=64, ff=256", "type": "Transformer"},
    {"name": "AttentionLSTM", "key": "lstm", "icon": "bi bi-layers",
     "desc": "LSTM with temporal attention mechanism for sequence modeling.",
     "arch": "2 layers, hidden=128, 4 attn heads", "type": "RNN"},
    {"name": "MomentumTransformer", "key": "momentum", "icon": "bi bi-lightning",
     "desc": "Transformer designed for momentum factor extraction and prediction.",
     "arch": "2 layers, 4 heads, d=64, ff=256", "type": "Transformer"},
    {"name": "PPOAgent", "key": "ppo", "icon": "bi bi-robot",
     "desc": "Proximal Policy Optimization reinforcement learning agent for trading.",
     "arch": "2 layers, hidden=128, clip=0.2", "type": "RL"},
    # Zug 37 Models
    {"name": "CausalDiscoveryTransformer", "key": "causal", "icon": "bi bi-diagram-2",
     "desc": "Transformer with a learned causal adjacency matrix for feature interactions.",
     "arch": "n layers, n heads, d_model, d_ff", "type": "Transformer"},
    {"name": "SchrodingerTransformer", "key": "schrodinger", "icon": "bi bi-brilliance",
     "desc": "Quantum-inspired transformer with parallel regime branches.",
     "arch": "n layers, n heads, n regimes", "type": "Transformer"},
    {"name": "TopologicalAttentionNetwork", "key": "topological", "icon": "bi bi-bezier2",
     "desc": "Topological Data Analysis-inspired transformer with Betti-0 features.",
     "arch": "n layers, n heads, n scales", "type": "Transformer"},
    {"name": "HamiltonianNeuralODE", "key": "hamiltonian", "icon": "bi bi-infinity",
     "desc": "Hamiltonian-inspired neural ODE using discrete symplectic leapfrog integration.",
     "arch": "leapfrog steps, d_model (even)", "type": "ODE"},
    {"name": "EntropicPortfolioDiffusion", "key": "diffusion", "icon": "bi bi-cloud-haze2",
     "desc": "Diffusion-inspired model with iterative denoising refinement.",
     "arch": "n layers, n heads, n diffusion steps", "type": "Diffusion"},
    {"name": "AdversarialRegimeModel", "key": "adversarial", "icon": "bi bi-shield-shaded",
     "desc": "GAN-inspired model with generator, discriminator, and regime classifier.",
     "arch": "n layers, n heads, n regimes", "type": "GAN"},
]

# Trainable model options for the dropdown (excludes PPO which has a separate training pipeline)
_TRAIN_MODEL_OPTIONS = [
    {"label": "All Models", "value": "all"},
    {"label": "Transformer", "value": "transformer"},
    {"label": "ITransformer", "value": "itransformer"},
    {"label": "LSTM", "value": "lstm"},
    {"label": "Momentum", "value": "momentum"},
    {"label": "Causal", "value": "causal"},
    {"label": "Schrödinger", "value": "schrodinger"},
    {"label": "Topological", "value": "topological"},
    {"label": "Hamiltonian", "value": "hamiltonian"},
    {"label": "Diffusion", "value": "diffusion"},
    {"label": "Adversarial", "value": "adversarial"},
]

# File-based state for cross-worker training tracking
_TRAIN_DIR = Path(tempfile.gettempdir()) / "quant_train_run"
_TRAIN_DIR.mkdir(exist_ok=True)
_TRAIN_PID_FILE = _TRAIN_DIR / "pid"
_TRAIN_LOG_FILE = _TRAIN_DIR / "output.log"
_TRAIN_RC_FILE = _TRAIN_DIR / "rc"


def _get_checkpoint_info(model_key):
    """Get checkpoint file info without loading torch."""
    info = {"exists": False}
    for suffix in ["_latest.pt", "_model.pt"]:
        path = _CHECKPOINT_DIR / f"{model_key}{suffix}"
        if path.exists():
            stat = path.stat()
            info["exists"] = True
            info["path"] = str(path.name)
            info["size_mb"] = stat.st_size / (1024 * 1024)
            info["modified"] = datetime.fromtimestamp(stat.st_mtime)
            break
    return info


def _load_training_metrics():
    """Load training_metrics.json, normalizing both old (float) and new (dict) formats."""
    metrics_path = _CHECKPOINT_DIR / "training_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    # Normalize: old format is {model: float}, new format is {model: dict}
    result = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            result[k] = v
        else:
            result[k] = {"directional_accuracy": float(v)}
    return result


def _model_cards():
    """Build model registry cards with checkpoint status and training performance."""
    cards = []
    training_metrics = _load_training_metrics()

    for m in MODEL_REGISTRY:
        ckpt = _get_checkpoint_info(m["key"])
        metrics = training_metrics.get(m["key"], {})

        # Status badge
        if ckpt["exists"]:
            status = html.Span("Trained", className="badge-pill profit",
                               style={"fontSize": "0.65rem"})
            ckpt_info = html.Div([
                html.Span(f"{ckpt['size_mb']:.1f} MB", className="badge-pill mono",
                          style={"fontSize": "0.65rem"}),
                html.Span(ckpt["modified"].strftime("%Y-%m-%d %H:%M"), className="badge-pill",
                          style={"fontSize": "0.65rem"}),
            ], style={"display": "flex", "gap": "0.3rem", "marginTop": "0.3rem"})
        else:
            status = html.Span("Not Trained", className="badge-pill loss",
                               style={"fontSize": "0.65rem"})
            ckpt_info = html.Div()

        # Training performance badges (only if metrics exist for this model)
        perf_info = html.Div()
        if metrics and ckpt["exists"]:
            perf_badges = []
            acc = metrics.get("directional_accuracy")
            if acc is not None:
                acc_class = "badge-pill profit" if acc >= 0.5 else "badge-pill loss"
                perf_badges.append(html.Span(
                    f"Acc {acc:.1%}", className=acc_class,
                    style={"fontSize": "0.65rem"},
                ))
            val_loss = metrics.get("best_val_loss")
            if val_loss is not None:
                perf_badges.append(html.Span(
                    f"Loss {val_loss:.4f}", className="badge-pill mono",
                    style={"fontSize": "0.65rem"},
                ))
            elapsed = metrics.get("elapsed")
            if elapsed is not None:
                perf_badges.append(html.Span(
                    f"{elapsed:.0f}s", className="badge-pill mono",
                    style={"fontSize": "0.65rem"},
                ))
            n_params = metrics.get("n_params")
            if n_params is not None:
                if n_params >= 1_000_000:
                    param_str = f"{n_params / 1_000_000:.1f}M params"
                elif n_params >= 1_000:
                    param_str = f"{n_params / 1_000:.1f}K params"
                else:
                    param_str = f"{n_params} params"
                perf_badges.append(html.Span(
                    param_str, className="badge-pill mono",
                    style={"fontSize": "0.65rem"},
                ))
            if perf_badges:
                perf_info = html.Div(
                    perf_badges,
                    style={"display": "flex", "gap": "0.3rem", "flexWrap": "wrap", "marginTop": "0.3rem"},
                )

        cards.append(html.Div([
            html.Div([
                html.Div([
                    html.I(className=m["icon"], style={"fontSize": "1.1rem", "color": C["accent"]}),
                    html.Span(m["name"], style={
                        "fontWeight": "700", "color": "var(--text-strong)", "fontSize": "0.88rem",
                    }),
                    status,
                ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
                html.P(m["desc"], style={
                    "color": "var(--text-weak)", "fontSize": "0.74rem",
                    "lineHeight": "1.45", "margin": "0.5rem 0 0.4rem",
                }),
                html.Div([
                    html.Span(m["type"], className="badge-pill accent", style={"fontSize": "0.65rem"}),
                    html.Span(m["arch"], className="badge-pill mono", style={"fontSize": "0.65rem"}),
                ], style={"display": "flex", "gap": "0.3rem", "flexWrap": "wrap"}),
                ckpt_info,
                perf_info,
            ]),
        ], className="card-surface"))

    return cards


def _trade_by_model_chart(all_results):
    """Show trade outcomes across backtests."""
    if not all_results:
        return html.P("No backtest data.", style={"color": "var(--text-weak)",
                       "fontSize": "0.75rem", "textAlign": "center", "padding": "2rem"})

    bt_names = []
    returns = []
    sharpes = []
    trades_count = []
    win_rates = []

    for name, data in all_results.items():
        m = data.get("metrics", {})
        bt_names.append(name.replace("_", " ").title())
        returns.append(m.get("total_return", 0) * 100)
        sharpes.append(m.get("sharpe_ratio", 0))
        trades_count.append(int(m.get("total_trades", 0)))
        win_rates.append(m.get("win_rate", 0) * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bt_names, y=returns, name="Return (%)",
        marker_color=[C["profit"] if r >= 0 else C["loss"] for r in returns],
        hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(**config.PLOTLY_LAYOUT, showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(**config.PLOTLY_YAXIS, title="Return (%)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "300px"})


def _metrics_comparison_table(all_results):
    """Side-by-side metrics comparison table."""
    if not all_results:
        return html.Div()

    headers = [html.Th("Metric", style={"color": "var(--text-weak)", "textAlign": "left",
                                          "padding": "0.5rem 1rem", "fontSize": "0.7rem",
                                          "borderBottom": "1px solid var(--border-default)"})]
    for name in all_results:
        headers.append(html.Th(name.replace("_", " ").title(),
                               style={"color": "var(--text-strong)", "textAlign": "right",
                                      "padding": "0.5rem 1rem", "fontSize": "0.7rem",
                                      "fontFamily": "var(--font-mono)",
                                      "borderBottom": "1px solid var(--border-default)"}))

    compare_metrics = [
        ("total_return", "Total Return", ".2%"),
        ("sharpe_ratio", "Sharpe", ".2f"),
        ("max_drawdown", "Max DD", ".2%"),
        ("total_trades", "Trades", ".0f"),
        ("win_rate", "Win Rate", ".1%"),
        ("profit_factor", "Profit Factor", ".2f"),
    ]

    rows = []
    for key, label, fmt in compare_metrics:
        cells = [html.Td(label, style={"color": "var(--text-weak)", "padding": "0.4rem 1rem",
                                        "fontSize": "0.73rem",
                                        "borderBottom": "1px solid var(--border-default)"})]
        vals = [data.get("metrics", {}).get(key, 0) for data in all_results.values()]
        best = max(vals) if key not in ("max_drawdown",) else min(vals)

        for val in vals:
            formatted = f"{val:{fmt}}"
            style = {"fontFamily": "var(--font-mono)", "padding": "0.4rem 1rem",
                     "fontSize": "0.73rem", "textAlign": "right",
                     "borderBottom": "1px solid var(--border-default)"}
            if val == best and len(vals) > 1:
                style["color"] = "var(--status-profit)"
                style["fontWeight"] = "600"
            else:
                style["color"] = "var(--text-default)"
            cells.append(html.Td(formatted, style=style))
        rows.append(html.Tr(cells))

    return html.Table([html.Thead(html.Tr(headers)), html.Tbody(rows)],
                      style={"width": "100%", "borderCollapse": "collapse"})


def _form_input(label, component, hint=None, flex=1):
    """Wrap a form input with a label and optional hint."""
    children = [
        html.Label(label, className="bt-form-label"),
        component,
    ]
    if hint:
        children.append(html.Span(hint, className="bt-form-hint"))
    return html.Div(children, style={"flex": str(flex), "minWidth": "140px"})


# ---------------------------------------------------------------------------
# File-based IPC helpers (same pattern as backtest.py)
# ---------------------------------------------------------------------------

def _train_is_running():
    """Check if a training process is currently running via PID file."""
    if not _TRAIN_PID_FILE.exists():
        return False
    try:
        pid = int(_TRAIN_PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def _train_read_log():
    """Read the current training log from file."""
    if _TRAIN_LOG_FILE.exists():
        try:
            return _TRAIN_LOG_FILE.read_text(errors="replace")
        except OSError:
            return ""
    return ""


def _train_read_rc():
    """Read the return code file. Returns None if not finished."""
    if _TRAIN_RC_FILE.exists():
        try:
            return int(_TRAIN_RC_FILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


# ---------------------------------------------------------------------------
# Static layout
# ---------------------------------------------------------------------------

layout = html.Div([
    dcc.Interval(id="models-refresh", interval=config.REFRESH_INTERVAL_MS, n_intervals=0),
    html.Div(id="models-content"),

    # Train Models form (static — outside the refresh callback to keep component IDs stable)
    html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Train Models", className="section-title"),
        ], className="section-header"),
        html.Div([
            # Row 1: Model, Assets, Interval
            html.Div([
                _form_input("Model", dcc.Dropdown(
                    id="train-run-model",
                    options=_TRAIN_MODEL_OPTIONS,
                    value="all",
                    clearable=False,
                    className="once-dropdown",
                    style={"width": "100%"},
                )),
                _form_input("Assets", dcc.Dropdown(
                    id="train-run-assets",
                    options=ASSET_OPTIONS,
                    value=["AAPL"],
                    multi=True,
                    placeholder="Select assets...",
                    searchable=True,
                    className="once-dropdown",
                    style={"width": "100%"},
                ), hint="Search or select multiple", flex=2),
                _form_input("Interval", dcc.Dropdown(
                    id="train-run-interval",
                    options=[
                        {"label": "1 min", "value": "1m"},
                        {"label": "5 min", "value": "5m"},
                        {"label": "15 min", "value": "15m"},
                        {"label": "1 hour", "value": "1h"},
                        {"label": "1 day", "value": "1d"},
                    ],
                    value="1d",
                    clearable=False,
                    className="once-dropdown",
                    style={"width": "100%"},
                )),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),

            # Divider
            html.Div(className="bt-form-divider"),

            # Row 2: Days, Epochs, Batch Size
            html.Div([
                _form_input("Training Days", dcc.Input(
                    id="train-run-days",
                    type="number",
                    value=365,
                    min=7,
                    step=1,
                    className="bt-form-input",
                ), hint="Calendar days of data"),
                _form_input("Epochs", dcc.Input(
                    id="train-run-epochs",
                    type="number",
                    value=100,
                    min=1,
                    step=10,
                    className="bt-form-input",
                )),
                _form_input("Batch Size", dcc.Input(
                    id="train-run-batch",
                    type="number",
                    value=64,
                    min=8,
                    step=8,
                    className="bt-form-input",
                )),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),

            # Divider
            html.Div(className="bt-form-divider"),

            # Run button + status
            html.Div([
                html.Div(id="train-run-status"),
                html.Button(
                    [
                        html.I(className="bi bi-cpu", style={"fontSize": "0.8rem"}),
                        "Train",
                    ],
                    id="train-run-btn",
                    n_clicks=0,
                    style={
                        "fontFamily": "var(--font-sans)", "fontSize": "0.78rem", "fontWeight": "700",
                        "padding": "0.6rem 1.6rem", "minHeight": "38px",
                        "borderRadius": "var(--radius-md)",
                        "border": "1px solid rgba(59, 130, 246, 0.4)",
                        "background": "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                        "color": "#fff", "cursor": "pointer",
                        "display": "inline-flex", "alignItems": "center", "gap": "0.4rem",
                        "letterSpacing": "0.01em",
                    },
                ),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "gap": "1rem"}),

            # Output log
            html.Pre(id="train-run-log", style={
                "fontFamily": "var(--font-mono)", "fontSize": "0.68rem",
                "color": C["text"], "padding": "0 0.8rem",
                "background": C["bg_overlay"], "borderRadius": "var(--radius-sm)",
                "border": f"1px solid {C['border']}",
                "whiteSpace": "pre-wrap",
            }),

            # Interval to poll for training completion
            dcc.Interval(id="train-run-poll", interval=2000, disabled=True),
        ], className="bt-run-card"),
    ], className="section"),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("models-content", "children"),
    Input("models-refresh", "n_intervals"),
)
def update_models(_n):
    all_results = loader.load_all_backtest_results()

    # Summary KPIs
    trained = sum(1 for m in MODEL_REGISTRY if _get_checkpoint_info(m["key"])["exists"])
    total_bt = len(all_results)

    from quant.config.settings import get_device
    device = get_device()
    device_label = str(device).upper()

    kpis = html.Div([
        create_kpi_card("Models", f"{len(MODEL_REGISTRY)}", "accent"),
        create_kpi_card("Trained", f"{trained}/{len(MODEL_REGISTRY)}",
                        "profit" if trained == len(MODEL_REGISTRY) else "warning"),
        create_kpi_card("Backtests", f"{total_bt}"),
        create_kpi_card("Device", device_label,
                        "accent" if device.type != "cpu" else ""),
    ], className="kpi-grid")

    # Model registry
    registry = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Div([
                html.Span("Model Registry", className="section-title"),
                html.Span(f"{len(MODEL_REGISTRY)} models", className="section-count"),
            ], style={"display": "flex", "flexDirection": "column", "gap": "0.1rem"}),
        ], className="section-header"),
        html.Div(_model_cards(), className="grid-auto"),
    ], className="section")

    # Backtest performance comparison
    comparison = html.Div()
    if all_results:
        comparison = html.Div([
            html.Div([
                html.Div([
                    html.Div(className="section-indicator"),
                    html.Span("Backtest Performance", className="section-title"),
                ], className="section-header"),
                html.Div(_trade_by_model_chart(all_results), className="chart-card"),
            ]),
            html.Div([
                html.Div([
                    html.Div(className="section-indicator"),
                    html.Span("Metrics Comparison", className="section-title"),
                ], className="section-header"),
                html.Div(_metrics_comparison_table(all_results), className="card-static",
                         style={"padding": "0.5rem 0", "overflow": "auto"}),
            ]),
        ], className="grid-2")

    return html.Div([kpis, registry, comparison])


# ---------------------------------------------------------------------------
# Train Model callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("train-run-poll", "disabled"),
    Output("train-run-status", "children"),
    Output("train-run-log", "className"),
    Output("train-run-log", "children"),
    Output("train-run-btn", "disabled"),
    Input("train-run-btn", "n_clicks"),
    State("train-run-model", "value"),
    State("train-run-assets", "value"),
    State("train-run-interval", "value"),
    State("train-run-days", "value"),
    State("train-run-epochs", "value"),
    State("train-run-batch", "value"),
    prevent_initial_call=True,
)
def start_training(n_clicks, model, assets, interval, days, epochs, batch_size):
    if not n_clicks or not all([model, assets, interval, days, epochs, batch_size]):
        return no_update, no_update, no_update, no_update, no_update

    if _train_is_running():
        status = html.Span([
            html.Span(className="status-dot", style={"background": "var(--status-warning)"}),
            "Training is already running.",
        ], className="bt-status-badge running")
        return no_update, status, no_update, no_update, no_update

    assets_str = ",".join(assets) if isinstance(assets, list) else assets
    script = _PROJECT_ROOT / "scripts" / "train_model.py"

    # Clean up previous run files
    for f in (_TRAIN_PID_FILE, _TRAIN_LOG_FILE, _TRAIN_RC_FILE):
        f.unlink(missing_ok=True)

    # Launch subprocess
    log_fh = open(_TRAIN_LOG_FILE, "w")
    proc = subprocess.Popen(
        [
            sys.executable, str(script),
            "--model", model,
            "--assets", assets_str,
            "--interval", interval,
            "--days", str(int(days)),
            "--epochs", str(int(epochs)),
            "--batch-size", str(int(batch_size)),
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(_PROJECT_ROOT),
        start_new_session=True,
    )

    _TRAIN_PID_FILE.write_text(str(proc.pid))

    def _wait():
        rc = proc.wait()
        log_fh.close()
        _TRAIN_RC_FILE.write_text(str(rc))

    threading.Thread(target=_wait, daemon=True).start()

    status = html.Span([
        html.Span(className="status-dot", style={
            "background": "var(--status-warning)",
            "boxShadow": "0 0 6px var(--status-warning)",
        }),
        "Running...",
    ], className="bt-status-badge running")

    return False, status, "visible", "Starting training...\n", True


@callback(
    Output("train-run-status", "children", allow_duplicate=True),
    Output("train-run-log", "children", allow_duplicate=True),
    Output("train-run-poll", "disabled", allow_duplicate=True),
    Output("train-run-btn", "disabled", allow_duplicate=True),
    Output("models-content", "children", allow_duplicate=True),
    Input("train-run-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_training(_n):
    log_text = _train_read_log()

    if _train_is_running():
        status = html.Span([
            html.Span(className="status-dot", style={
                "background": "var(--status-warning)",
                "boxShadow": "0 0 6px var(--status-warning)",
            }),
            "Running...",
        ], className="bt-status-badge running")
        return status, log_text or "Starting training...\n", no_update, True, no_update

    rc = _train_read_rc()

    if rc is None and not _TRAIN_PID_FILE.exists():
        return no_update, no_update, True, False, no_update

    if rc is None:
        return no_update, log_text, no_update, True, no_update

    # Clear loader cache so model cards refresh with new checkpoint timestamps
    loader.clear_cache()

    if rc == 0:
        status = html.Span([
            html.I(className="bi bi-check-circle-fill"),
            "Complete",
        ], className="bt-status-badge complete")
    else:
        status = html.Span([
            html.I(className="bi bi-exclamation-triangle-fill"),
            f"Failed (exit {rc})",
        ], className="bt-status-badge failed")

    # Rebuild models content immediately so cards reflect new checkpoint status
    return status, log_text, True, False, update_models(0)
