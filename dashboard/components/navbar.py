"""Top header with pill-style navigation — Once UI style."""

from dash import html, dcc, callback, Output, Input

NAV_ITEMS = [
    {"label": "Overview", "href": "/", "icon": "bi bi-grid-1x2"},
    {"label": "Positions", "href": "/positions", "icon": "bi bi-wallet2"},
    {"label": "Backtest", "href": "/backtest", "icon": "bi bi-clock-history"},
    {"label": "Models", "href": "/models", "icon": "bi bi-cpu"},
    {"label": "Risk", "href": "/risk", "icon": "bi bi-shield-check"},
    {"label": "System", "href": "/system", "icon": "bi bi-gear"},
]


def create_header():
    """Create the top header with title, nav pills, and status badge."""
    nav_pills = html.Div(
        [
            dcc.Link(
                [html.I(className=item["icon"]), html.Span(item["label"])],
                href=item["href"],
                className="nav-pill",
                id=f"nav-{item['label'].lower()}",
            )
            for item in NAV_ITEMS
        ],
        className="nav-pills-container",
    )

    status_badge = html.Div(
        [
            html.Div(className="status-dot"),
            html.Span("System Active"),
        ],
        className="system-status",
    )

    return html.Header(
        [
            html.Div(
                [
                    html.H1("Quant 2.0", className="header-title"),
                    html.Div("Self-Learning Trading System \u2022 v2.0.0", className="header-subtitle"),
                ],
                className="header-left",
            ),
            html.Div(
                [nav_pills, status_badge],
                className="header-right",
            ),
        ],
        className="top-header",
    )


@callback(
    [Output(f"nav-{item['label'].lower()}", "className") for item in NAV_ITEMS],
    Input("url", "pathname"),
)
def update_active_nav(pathname):
    """Set active class on the current page's nav pill."""
    classes = []
    for item in NAV_ITEMS:
        if pathname == item["href"] or (pathname and item["href"] != "/" and pathname.startswith(item["href"])):
            classes.append("nav-pill active")
        else:
            classes.append("nav-pill")
    return classes
