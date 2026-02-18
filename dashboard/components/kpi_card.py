"""Once UI style KPI card component."""

from dash import html


def create_kpi_card(title, value, color_class="", subtitle=None):
    """Reusable KPI metric card with Once UI styling.

    Args:
        title: Card label text (uppercase).
        value: Main display value.
        color_class: CSS class for value color (profit, loss, warning, accent).
        subtitle: Optional small text below value.
    """
    children = [
        html.Div(title, className="kpi-label"),
        html.Div(value, className=f"kpi-value {color_class}"),
    ]
    if subtitle:
        children.append(
            html.Div(subtitle, style={
                "fontFamily": "var(--font-mono)",
                "fontSize": "0.65rem",
                "color": "var(--text-disabled)",
                "marginTop": "0.15rem",
            })
        )
    return html.Div(children, className="kpi-card")
