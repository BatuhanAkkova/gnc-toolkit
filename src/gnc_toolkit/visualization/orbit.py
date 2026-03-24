import numpy as np
import plotly.graph_objects as go


def plot_orbit_3d(r_eci, r_earth=True, title="3D Orbit Visualization"):
    """
    Plots a 3D orbit trajectory around a spherical Earth mesh using plotly.

    Parameters
    ----------
        r_eci (numpy.ndarray or list): Shape (N, 3) or List of 3-element lists/arrays representing position vectors in ECI [m].
        r_earth (bool): Whether to plot a spherical Earth model. Default is True.
        title (str): Title for the plot.

    Returns
    -------
        plotly.graph_objects.Figure: The plotly figure object.
    """
    r_eci = np.array(r_eci)

    if r_eci.ndim != 2 or r_eci.shape[1] != 3:
        raise ValueError("r_eci must be a 2D array of shape (N, 3)")

    fig = go.Figure()

    # 1. Plot Earth
    if r_earth:
        # Earth Radius in SI Units [m]
        R_Earth = 6378137.0

        # Create meshgrid for sphere
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x = R_Earth * np.outer(np.cos(u), np.sin(v))
        y = R_Earth * np.outer(np.sin(u), np.sin(v))
        z = R_Earth * np.outer(np.ones(np.size(u)), np.cos(v))

        # Add Surface
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Blues",
                opacity=0.3,  # Translucent so lines passing inside/behind are visible if needed
                showscale=False,
                name="Earth",
                hoverinfo="skip",
            )
        )

    # 2. Plot Orbit Path
    fig.add_trace(
        go.Scatter3d(
            x=r_eci[:, 0],
            y=r_eci[:, 1],
            z=r_eci[:, 2],
            mode="lines",
            line=dict(color="darkred", width=4),
            name="Orbit Path",
        )
    )

    # 3. Add Markers for Start and End Points
    fig.add_trace(
        go.Scatter3d(
            x=[r_eci[0, 0]],
            y=[r_eci[0, 1]],
            z=[r_eci[0, 2]],
            mode="markers",
            marker=dict(size=6, color="green", symbol="circle"),
            name="Start",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[r_eci[-1, 0]],
            y=[r_eci[-1, 1]],
            z=[r_eci[-1, 2]],
            mode="markers",
            marker=dict(size=6, color="orange", symbol="circle"),
            name="Current Position",
        )
    )

    # 4. Set Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X [m]", gridcolor="gray", showbackground=False),
            yaxis=dict(title="Y [m]", gridcolor="gray", showbackground=False),
            zaxis=dict(title="Z [m]", gridcolor="gray", showbackground=False),
            aspectmode="data",  # Keeps 1:1:1 proportions
        ),
        margin=dict(r=10, l=10, b=10, t=40),
    )

    return fig
