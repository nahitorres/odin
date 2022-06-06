import cv2
import ipywidgets as widgets
from ipyleaflet import Map, LocalTileLayer, ImageOverlay, projections
from ipyleaflet import WidgetControl
from odin.utils.env import get_leaflet_url


config = {"center":(50, 50),
         "min_zoom": 2,
         "size":'600px'}

sizemap = {'400px': 0, '500px': -12, '600px': -25, '700px': -37, '800px': -50, '900px': -62}

def get_image_container_zoom():
    image_container = Map(center = config["center"], min_zoom=config["min_zoom"], zoom=config["min_zoom"], dragging=True,
                          zoom_control=True, box_zoom=True, double_click_zoom=True, bounce_at_zoom_limits=False,
                          layers=[LocalTileLayer(path='../classes/white.png')], layout=dict(width=config["size"], height=config["size"]),
                          crs=projections.Simple)
    
    button = widgets.Button(    
        disabled=False,
        button_style='', 
        icon='arrows-alt', 
        layout= widgets.Layout(width='35px', height='30px')
    )
    def function(b):
        center_image_on_map(image_container)
    
    button.on_click(function)
    recenter_control = WidgetControl(widget = button, position = 'topleft')
    image_container.add_control(recenter_control)
    
    return image_container

def center_image_on_map(image_container):
    image_container.center = config["center"]
    image_container.zoom = config["min_zoom"]
    
def get_image_overlay(img_path):
    
    # We need to create an ImageOverlay for each image to show,
    # and set the appropriate bounds based  on the image size
    
    im = Image.open(img_path)
    w, h = im.size

    max_v = 100

    offset_h = sizemap[config["size"]]
    offset_w = sizemap[config["size"]]
    hh = max_v - offset_h*2
    ww = max_v - offset_w*2

    if h > w:
        ww = int(w * hh / h)
        offset_w = (max_v - ww) / 2
    else:
        hh = int(h * ww / w)
        offset_h = (max_v - hh) / 2

    img_ov = ImageOverlay(url=get_leaflet_url(im), bounds=((offset_h, offset_w), (hh + offset_h, ww+offset_w)))
    return img_ov, h, w, hh, ww, offset_h, offset_w


def show_new_image(image_container, image_path):
    for layer in image_container.layers:
        image_container.remove_layer(layer)
    image_layer, h, w, hh, ww, off_h, off_w = get_image_overlay(image_path)
    image_container.add_layer(image_layer)
    image_container.center = config["center"]
    image_container.zoom = config["min_zoom"]