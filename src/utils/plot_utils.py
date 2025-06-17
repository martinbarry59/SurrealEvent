import plotly
import numpy as np

def plotly_TSNE(tsne, labels, video_label, name):
    ## nice plotly with labels on hover
    fig = plotly.graph_objs.Figure()
    unique_labels = np.unique(video_label)
    for i, label in enumerate(unique_labels):
        idx = video_label == label
        fig.add_trace(
            plotly.graph_objs.Scatter(
                x=tsne[idx, :][:, 0],
                y=tsne[idx, :][:, 1],
                mode="markers",
                name=label,
                text=labels[idx],
                hoverinfo="text",
            )
        )

    ## save the htmtl plot
    
    fig.update_layout(title=name)
    plotly.offline.plot(fig, filename=name + ".html")