import numpy as np
import plotly.graph_objects as go

def plot_attitude_sphere(vectors, title="Attitude Sphere Visualization"):
    """
    Plots vectors on a unit sphere to visualize pointing direction or attitude evolution.

    Parameters:
        vectors (numpy.ndarray or list): Shape (N, 3) representing unit vectors in inertial/reference frame.
        title (str): Title for the plot.

    Returns:
        plotly.graph_objects.Figure: The plotly figure object.
    """
    vectors = np.array(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
         raise ValueError("vectors must be a 2D array of shape (N, 3)")

    # Normalize vectors to ensure they are on the unit sphere
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # Avoid division by zero
    vectors_norm = vectors / norms

    fig = go.Figure()

    # 1. Plot Unit Sphere Mesh (Wireframe / Translucent)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Greys',
        opacity=0.15,
        showscale=False,
        name="Unit Sphere",
        hoverinfo='skip'
    ))

    # 2. Plot the Trace of the vectors
    fig.add_trace(go.Scatter3d(
        x=vectors_norm[:, 0],
        y=vectors_norm[:, 1],
        z=vectors_norm[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name="Pointing Trace"
    ))

    # 3. Add Line from Center to Current Vector (e.g., last point)
    current_vec = vectors_norm[-1]
    fig.add_trace(go.Scatter3d(
        x=[0, current_vec[0]],
        y=[0, current_vec[1]],
        z=[0, current_vec[2]],
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=4),
        name="Current Vector"
    ))

    # 4. Add Starting Point Marker
    fig.add_trace(go.Scatter3d(
        x=[vectors_norm[0, 0]],
        y=[vectors_norm[0, 1]],
        z=[vectors_norm[0, 2]],
        mode='markers',
        marker=dict(size=6, color='green'),
        name="Start"
    ))

    # 5. Set Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", range=[-1.1, 1.1], gridcolor='lightgray'),
            yaxis=dict(title="Y", range=[-1.1, 1.1], gridcolor='lightgray'),
            zaxis=dict(title="Z", range=[-1.1, 1.1], gridcolor='lightgray'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1) # Keep it spherical
        ),
        margin=dict(r=10, l=10, b=10, t=40)
    )

    return fig
