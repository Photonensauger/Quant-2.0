import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input

from dashboard import config
from dashboard.components.navbar import create_header

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.icons.BOOTSTRAP],
    title=config.APP_TITLE,
    suppress_callback_exceptions=True,
)

server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                create_header(),
                html.Div(dash.page_container, id="page-content"),
                html.Footer(
                    html.P(
                        ["Quant 2.0 ", html.Span("Trading Dashboard"), " \u2014 v2.0.0"],
                    ),
                    className="site-footer",
                ),
            ],
            className="page-wrapper",
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
