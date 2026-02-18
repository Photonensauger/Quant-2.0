"""Sortable/filterable trade table using Dash DataTable."""

from dash import dash_table

from dashboard.config import COLORS


def create_trade_table(trade_log):
    """Trade log DataTable with sort, filter, pagination, and P&L color coding."""
    if not trade_log:
        return dash_table.DataTable(
            data=[],
            columns=[{"name": c, "id": c} for c in ["#", "Asset", "Side", "Entry Time", "Exit Time", "Qty", "Entry $", "Exit $", "P&L"]],
            page_size=25,
            style_table={"overflowX": "auto"},
        )

    records = []
    for i, trade in enumerate(trade_log, 1):
        pnl = trade.get("pnl", 0)
        records.append({
            "#": i,
            "Asset": trade.get("asset", trade.get("symbol", "N/A")),
            "Side": trade.get("side", "N/A").upper(),
            "Entry Time": str(trade.get("entry_time", "N/A"))[:16],
            "Exit Time": str(trade.get("exit_time", "N/A"))[:16],
            "Qty": round(trade.get("qty", 0), 2),
            "Entry $": round(trade.get("entry_price", 0), 2),
            "Exit $": round(trade.get("exit_price", 0), 2),
            "P&L": round(pnl, 2),
        })

    columns = [
        {"name": "#", "id": "#", "type": "numeric"},
        {"name": "Asset", "id": "Asset", "type": "text"},
        {"name": "Side", "id": "Side", "type": "text"},
        {"name": "Entry Time", "id": "Entry Time", "type": "text"},
        {"name": "Exit Time", "id": "Exit Time", "type": "text"},
        {"name": "Qty", "id": "Qty", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Entry $", "id": "Entry $", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "Exit $", "id": "Exit $", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "P&L", "id": "P&L", "type": "numeric", "format": {"specifier": "+,.2f"}},
    ]

    return dash_table.DataTable(
        data=records,
        columns=columns,
        sort_action="native",
        filter_action="native",
        page_size=25,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "0.55rem 0.85rem",
            "fontFamily": "var(--font-mono)",
            "fontSize": "0.72rem",
            "border": "none",
            "borderBottom": f"1px solid {COLORS['border']}",
        },
        style_header={
            "backgroundColor": COLORS["bg_elevated"],
            "color": COLORS["text_weak"],
            "fontFamily": "var(--font-sans)",
            "fontSize": "0.68rem",
            "fontWeight": "600",
            "textTransform": "uppercase",
            "letterSpacing": "0.04em",
            "borderBottom": f"1px solid {COLORS['border']}",
        },
        style_data={
            "backgroundColor": COLORS["bg_surface"],
            "color": COLORS["text"],
        },
        style_data_conditional=[
            {
                "if": {"filter_query": "{P&L} >= 0", "column_id": "P&L"},
                "color": COLORS["profit"],
                "fontWeight": "600",
            },
            {
                "if": {"filter_query": "{P&L} < 0", "column_id": "P&L"},
                "color": COLORS["loss"],
                "fontWeight": "600",
            },
            {
                "if": {"filter_query": '{Side} = "LONG"', "column_id": "Side"},
                "color": COLORS["profit"],
            },
            {
                "if": {"filter_query": '{Side} = "SHORT"', "column_id": "Side"},
                "color": COLORS["loss"],
            },
        ],
        style_filter={
            "backgroundColor": COLORS["bg_elevated"],
            "color": COLORS["text_strong"],
            "fontFamily": "var(--font-mono)",
            "fontSize": "0.72rem",
        },
    )
