import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"
import webbrowser
import os


def plot_VE(df_dict, show=True):
    fig = go.Figure()
    fig.update_layout(title="Minute Volume Comparison")
    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="VE")

    for name, df in df_dict.items():
        x_col = "breathTime" if "breathTime" in df.columns else "Time"
        fig = fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df["VE"],
            name=name
        ))
    if show:
        fig.show()
    return fig


def plot_VT(df_dict, show=True):
    fig = go.Figure()
    fig.update_layout(title="Tidal Volume Comparison")
    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="VT")

    for name, df in df_dict.items():
        x_col = "breathTime" if "breathTime" in df.columns else "Time"
        fig = fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df["VT"],
            name=name
        ))
    if show:
        fig.show()
    return fig


def plot_df_columns(df_dict, plottitle="", xtitle="", ytitle="", show=False, renderer = None):
    if renderer is not None:
        pio.renderers.default = renderer
    fig = go.Figure()
    fig.update_layout(title=plottitle)
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)

    for name, (df, xcol, ycol) in df_dict.items():
        y_val = df.index if ycol == "index" else df[ycol]
        fig = fig.add_trace(go.Scatter(
            x=df.index if xcol == "index" else df[xcol],
            y=df[ycol],
            name=name
        ))
    if show:
        fig.show()
    return fig

def create_subplots_w_raw(df_dict, raw_df_dict, plottitle="", xtitle="", ytitle1="", ytitle2 = "", show=False):
    fig = make_subplots(rows = 2, cols=1, shared_xaxes= True)
    fig.update_layout(title=plottitle)

    fig.update_xaxes(title_text = xtitle)


    # Add traces from df_dict
    for name, (df, xcol, ycol) in df_dict.items():
        fig = fig.add_trace(go.Scatter(
            x=df[xcol],
            y=df[ycol],
            name=name
        ), row = 1, col = 1)
    fig.update_yaxes(title_text = ytitle1, row = 1)

    # Add raw data on bottom subplot
    for name, (df, xcol, ycol) in raw_df_dict.items():
        fig = fig.add_trace(go.Scatter(
            x=df[xcol],
            y=df[ycol],
            name=name
        ), row = 2, col = 1)
    fig.update_yaxes(title_text = ytitle2, row = 2)

    if show:
        fig.show()
    return fig



def figures_to_html(figs, filename="dashboard.html", show=False):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
    if show:
        webbrowser.open('file://' + os.path.realpath(filename))


def create_spectrogram(data, fs, name):
    from scipy.signal import spectrogram
    freqs, bins, Sxx = spectrogram(data, 25)
    trace = [go.Heatmap(
        x=bins,
        y=freqs,
        z=10 * np.log10(Sxx),
        colorscale='Jet',
    )]
    layout = go.Layout(
        title=f"Spectrogram: {name}",
        yaxis=dict(title='Frequency'),  # x-axis label
        xaxis=dict(title='Time'),  # y-axis label
    )
    fig = go.Figure(data=trace, layout=layout)

    return fig


def figures_to_html(figs, filename="dashboard.html", show=False):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
    if show:
        webbrowser.open('file://' + os.path.realpath(filename))
