import numpy as np
import plotly.graph_objects as go

def plot_ground_track(latitudes, longitudes, times=None, title="Satellite Ground Track"):
    """
    Plots sub-satellite points on an Earth map to visualize the ground track.

    Parameters:
        latitudes (numpy.ndarray or list): Geodetic/Geocentric Latitudes in [deg] (from -90 to 90).
        longitudes (numpy.ndarray or list): Geodetic/Geocentric Longitudes in [deg] (from -180 to 180).
        times (numpy.ndarray or list, optional): Timestamps or seconds for the trajectory.
        title (str): Title for the plot.

    Returns:
        plotly.graph_objects.Figure: The plotly figure object.
    """
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)

    if latitudes.shape != longitudes.shape:
        raise ValueError("Latitudes and longitudes must have the same length.")

    fig = go.Figure()

    # 1. Plot GeoScatter trace
    
    hover_text = []
    if times is not None:
        times = np.array(times)
        for t, lat, lon in zip(times, latitudes, longitudes):
            hover_text.append(f"T: {t:.1f}s<br>Lat: {lat:.2f}°<br>Lon: {lon:.2f}°")
    else:
         for lat, lon in zip(latitudes, longitudes):
            hover_text.append(f"Lat: {lat:.2f}°<br>Lon: {lon:.2f}°")

    fig.add_trace(go.Scattergeo(
        lat=latitudes,
        lon=longitudes,
        mode='lines+markers',
        line=dict(width=2, color='darkorange'),
        marker=dict(size=4, color='darkorange', opacity=0.8),
        hovertext=hover_text,
        hoverinfo='text',
        name="Ground Track"
    ))

    # 2. Add Start and End Point Markers
    fig.add_trace(go.Scattergeo(
        lat=[latitudes[0]],
        lon=[longitudes[0]],
        mode='markers',
        marker=dict(size=8, color='green', symbol='circle'),
        name="Start"
    ))
    
    fig.add_trace(go.Scattergeo(
        lat=[latitudes[-1]],
        lon=[longitudes[-1]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='square'),
        name="End"
    ))

    # 3. Update Layout
    fig.update_layout(
        title=title,
        geo=dict(
            showland=True,
            showcoastlines=True,
            projection_type='equirectangular', # Standard 2D flat map projection
            coastlinecolor='gray',
            landcolor='white',
            lakecolor='lightblue',
            rivercolor='lightblue',
            showocean=True,
            oceancolor='aliceblue',
            lataxis=dict(range=[-90, 90], showgrid=True, gridcolor='lightgray'),
            lonaxis=dict(range=[-180, 180], showgrid=True, gridcolor='lightgray')
        ),
        margin=dict(r=0, l=0, b=0, t=40)
    )

    return fig
