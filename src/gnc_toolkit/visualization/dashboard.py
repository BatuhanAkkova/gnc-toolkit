import dash
from dash import dcc, html


def create_dashboard_app(figures, title="GNC Toolkit Mission Dashboard"):
    """
    Creates a Dash application to display multiple figures together.

    Parameters
    ----------
        figures (dict): Dictionary mapping { "Figure Title": plotly.graph_objects.Figure }
        title (str): Title for the dashboard webpage.

    Returns
    -------
        dash.Dash: The Dash application object.
    """
    app = dash.Dash(__name__, title=title)

    # Simple Flexbox layout for grid arrangement
    cards = []
    for name, fig in figures.items():
        cards.append(
            html.Div(
                [
                    html.H4(name, style={"margin-bottom": "5px", "color": "#333"}),
                    dcc.Graph(figure=fig),
                ],
                style={
                    "width": "45%",
                    "minWidth": "400px",
                    "margin": "15px",
                    "padding": "10px",
                    "backgroundColor": "white",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                },
            )
        )

    app.layout = html.Div(
        [
            html.Header(
                [
                    html.H1(title, style={"margin": "0", "fontSize": "28px"}),
                    html.P(
                        "Guidance, Navigation & Control Toolkit",
                        style={"margin": "5px 0 0 0", "opacity": "0.8"},
                    ),
                ],
                style={
                    "backgroundColor": "#2c3e50",
                    "color": "white",
                    "padding": "20px",
                    "textAlign": "center",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.2)",
                },
            ),
            html.Div(
                cards,
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "justifyContent": "space-around",
                    "padding": "20px",
                    "backgroundColor": "#ecf0f1",
                },
            ),
        ],
        style={
            "fontFamily": "Arial, sans-serif",
            "margin": "0",
            "backgroundColor": "#ecf0f1",
            "minHeight": "100vh",
        },
    )

    return app
