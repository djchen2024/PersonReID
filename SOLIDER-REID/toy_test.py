 




import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import cv2
import os
import numpy as np
from sklearn.manifold import TSNE
import base64


# Create dash app
app = dash.Dash(__name__)

# Set dog and cat images
dogImage = "https://www.iconexperience.com/_img/v_collection_png/256x256/shadow/dog.png"
catImage = "https://d2ph5fj80uercy.cloudfront.net/06/cat3602.jpg"
sample = "/local4TB/projects/dingjie/data/reid_CUSTOM_v1/query/0071_c3_00300.jpg"
encoded_image = base64.b64encode(open(sample, 'rb').read())


###################################################
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

max_img = 50
images = []
labels = []
X = X[:max_img, :]
Y = Y[:max_img]
for i in range(max_img):
    # img = cv2.imread(img_path_list[i])
    img = cv2.imread("/local4TB/projects/dingjie/data/reid_CUSTOM_v1/query/0071_c3_00300.jpg")
    print(img_path_list[i])
    print("Img shape: ", img.shape)
    images.append(img)
    labels.append(Y[i])
print("Show images: ", len(images))
print("Labels: ", labels)
 
# Flatten image matrices from (28,28) to (784,)
flattenend_images = [i.flatten() for i in images]
 
# t-SNE Outputs a 3 dimensional point for each image
# n_components = 2  # To get TSNE embedding with 2 dimensions
# tsne = TSNE(n_components)
# tsne_result = tsne.fit_transform(X)
tsne = TSNE(
    random_state = 123,
    n_components=3,
    verbose=0,
    perplexity=40,
    n_iter=300) \
    .fit_transform(X) 
#################################################





# Generate dataframe
df = pd.DataFrame(
   dict(
      x=[1, 2],
      y=[2, 4],
      images=[dogImage,sample],
   )
)

# Create scatter plot with x and y coordinates
fig = px.scatter(df, x="x", y="y",custom_data=["images"])

# Update layout and update traces
fig.update_layout(clickmode='event+select')
fig.update_traces(marker_size=20)

# Create app layout to show dash graph
app.layout = html.Div(
   [
      dcc.Graph(
         id="graph_interaction",
         figure=fig,
      ),
      # html.Img(id='image', src='')
      html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
   ]
)

# html callback function to hover the data on specific coordinates
@app.callback(
   Output('image', 'src'),
   Input('graph_interaction', 'hoverData'))
def open_url(hoverData):
   if hoverData:
      return hoverData["points"][0]["customdata"][0]
   else:
      raise PreventUpdate

if __name__ == '__main__':
   app.run_server(debug=True, port=1111)
