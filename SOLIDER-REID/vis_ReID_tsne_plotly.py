import io
import base64
import pickle
import gzip
import numpy as np
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image
from sklearn.manifold import TSNE
import cv2
import os
import plotly.express as px
import random
# from werkzeug.debug.tbtools import DebugTraceback # DJ @ /home/dingjie/miniconda3/envs/solider/lib/python3.10/site-packages/dash/dash.py

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url






# define images and labels
X = np.load("./vis/feat_array.npy") # do_streamed_inference from SOLIDER-REID/processor/processor.py
Y = np.load("./vis/pid_array.npy")
C = np.load("./vis/cid_array.npy")
f = open("./vis/path_list.txt", 'r')
img_path_list = f.readlines()
f.close()
print("Num of V1 images: ", len(img_path_list))
num_img, num_dim = X.shape
assert len(img_path_list)==num_img

max_img = 3000
images = []
labels = []
aspect_ratio = []
short = []
X = X[:max_img, :]
Y = Y[:max_img]
for i in range(max_img):
    img = cv2.imread(img_path_list[i][:-1])
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   #  print(img_path_list[i])
    # print("Img shape: ", img.shape)
    images.append(im_rgb)
    labels.append(Y[i])
    aspect_ratio.append(img.shape[0]/img.shape[1])
    if img.shape[0] < img.shape[1]:
        short.append(img.shape[0])
    else:
        short.append(img.shape[1])

# filter small aspect ratio images
min_aspect_ratio = 1.5
indices_small_aspect_ratio = [i for i, ar in enumerate(aspect_ratio) if ar < min_aspect_ratio]

min_short = 100
indices_small_short = [i for i, s in enumerate(short) if s < min_short]


# t-SNE Outputs a 3 dimensional point for each image
# n_components = 3  # To get TSNE embedding with 2 dimensions
# tsne = TSNE(n_components)
# tsne = tsne.fit_transform(X)
tsne = TSNE(
    random_state = 123,
    n_components=3,
    verbose=0,
    perplexity=60,
    n_iter=800) \
    .fit_transform(X) 



# breakpoint()

# Color for each digit
# color_map = {
#     0: "#E52B50",
#     1: "#9F2B68",
#     2: "#3B7A57",
#     3: "#3DDC84",
#     4: "#FFBF00",
#     5: "#915C83",
#     6: "#008000",
#     7: "#7FFFD4",
#     8: "#E9D66B",
#     9: "#007FFF",
# }
# colors = [color_map[l] for l in labels]
color_red = ["#FF0000"]
color_AliceBlue = ["#F0F8FF"]
color_alpha26 = px.colors.qualitative.Alphabet
color_dark24  = px.colors.qualitative.Dark24
color_light24 = px.colors.qualitative.Light24
color_vivid11 = px.colors.qualitative.Vivid
color_prism11 = px.colors.qualitative.Prism
color_list = color_alpha26+color_dark24+color_light24+color_vivid11+color_prism11
# random.shuffle(color_list)
# color_list = color_red + color_list
colors = [color_list[l] for l in labels]


# CASE 1 ----- filter out the images of small aspect ratio
# for i in indices_small_aspect_ratio:
#     colors[i] = color_AliceBlue[0]

# CASE 2 ----- filter out the images of short
# for i in indices_small_short:
#     colors[i] = color_AliceBlue[0]

# CASE 3 ----- filter out one sepcific cluster
# this_cluster_id = 18
# indices_other_clusters = [i for i, id in enumerate(labels) if id != this_cluster_id] 
# for i in indices_other_clusters:
#     colors[i] = color_AliceBlue[0]

# CASE 4 ----- filter out some sepcific clusters
# these_cluster_ids = [1, 2, 3, 4]
# indices_other_clusters = [i for i, id in enumerate(labels) if id not in these_cluster_ids] 
# for i in indices_other_clusters:
#     colors[i] = color_AliceBlue[0]

fig = go.Figure(data=[go.Scatter3d(
    x=tsne[:, 0],
    y=tsne[:, 1],
    z=tsne[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=colors,
    )
)])

fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)
fig.update_layout(
    autosize=False,
    width=1500,
    height=1200,
    scene=dict(
        xaxis=dict(range=[-50,50]),
        yaxis=dict(range=[-50,50]),
        zaxis=dict(range=[-50,50]),   
    )
)

app = JupyterDash(__name__) # ------------------------------------------------

app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
    ],
)

@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    im_matrix = images[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "100px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P("Individual ID: " + str(labels[num]), style={'font-weight': 'bold'})
        ])
    ]
    return True, bbox, children

print("Plot", len(images), "images ")


if __name__ == "__main__":
    app.run_server(debug=True)