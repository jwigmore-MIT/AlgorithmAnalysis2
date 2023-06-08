import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.signal import spectrogram

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

def make_time_trace(x, y, name, legend_group = None):
    return go.Scatter(
        x = x,
        y = y,
        name = name,
        legendgroup = legend_group,
    )
def make_spect_trace(data, fs= 25, name = "", legend_group = None):
    freqs, bins, Sxx = spectrogram(data, 25)
    trace = go.Heatmap(
        x=bins,
        y=freqs,
        z=10 * np.log10(Sxx),
        name=name,
        legendgroup = legend_group,
        colorscale='Jet',
    )
    return trace

def plot_time_spec(df_dict, show=False):
    fig = make_subplots(rows = 2*len(df_dict), cols=1, shared_xaxes= True)
    fig.update_layout(title="Time and Spectrogram Comparison")
    fig.update_xaxes(title_text="Time [s]")

    # Add traces from df_dict
    time_row = 1
    for i, (name, (df, xcol, ycol)) in enumerate(df_dict.items()):
        spec_row = time_row+1

        if xcol == "index":
            xvals = df.index
        else:
            xvals = df[xcol]
        fig = fig.add_trace(make_time_trace(xvals, df[ycol], name), row = time_row, col = 1)
        fig = fig.add_trace(make_spect_trace(df[ycol], name), row = spec_row, col = 1)
        time_row += 2
    fig.update_layout(height=500 * 2*len(df_dict),
                      legend_tracegroupgap=450,
                      )
    if show:
        fig.show()
    return fig

def create_subplots(df_dicts, plottitle="", xtitles=[], ytitles=[], subplot_titles = [], show=False):
    num_subplots = len(df_dicts)
    fig = make_subplots(rows = num_subplots, cols=1, shared_xaxes= True, subplot_titles = subplot_titles, vertical_spacing=0.05)
    fig.update_layout(title=plottitle)
    # Show legent on all subplots
    fig.update_layout(showlegend=True)
    for df_dict in df_dicts:
        n_row = df_dicts.index(df_dict) + 1
        for name, (df, xcol, ycol) in df_dict.items():
            if xcol == "index":
                xvals = df.index
            else:
                xvals = df[xcol]

            fig = fig.add_trace(go.Scatter(
                x= xvals,
                y=df[ycol],
                name=name,
                legendgroup = n_row,
            ), row = n_row, col = 1)
        fig.update_yaxes(title_text = ytitles[df_dicts.index(df_dict)], row = n_row)
        fig.update_xaxes(title_text = xtitles[df_dicts.index(df_dict)], row = n_row)
    # Increase figure size based on the number of subplots
    fig.update_layout(height=750*num_subplots,
                      legend_tracegroupgap=700,

                      )

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


def create_spectrogram_fig(data, fs = 25, name = ""):
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
