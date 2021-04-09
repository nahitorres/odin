import os
import cv2
import json
import glob
import time
import numpy as np
from PIL import Image
from datetime import datetime
import ipywidgets as widgets
from matplotlib import pyplot as plt
from IPython.display import clear_output
from ipyleaflet import Map,  DrawControl, ImageOverlay, LocalTileLayer, Rectangle, GeoJSON, Popup, projections
from odin.classes import strings as labels_str

class DatasetAnnotatorDetection:
    
    def __init__(self, classes, images, output_path, output_name, images_classes=None):
        # task classes
        self.__classes = classes
        self.__current_image_annotations = []
        self.__current_class_for_ann = []
        self.__output_path = output_path
        self.__output_name = output_name
        self.__file_path = os.path.join(self.__output_path, self.__output_name + ".json")
        self.__images = images
        # classes for each img
        self.__images_classes = images_classes
        self.__current_img = -1

        self.__image_id = -1
        self.__annotation_id = -1
        self.__selected_ann = None

        if os.path.exists(self.__file_path):
            self.__load_coco_file()
        else:
            self.__create_coco_format()

        self.__id_for_class = {cl['name']: cl['id'] for cl in self.__CATEGORIES}

        self.__create_btns()

        self.__progress = widgets.IntProgress(min=0, max=len(images), value=1, description='{}/{}'.format(1, len(images)))
        self.__progress.bar_style = 'info'
        self.__title_lbl = widgets.Label(value="title")
        self.__create_map()
        self.__create_classes_btns()
        self.__validation_show = widgets.HTML(value="")
        self.__map_classes = widgets.HBox(children=[self.__map, widgets.VBox(children=self.__classes_btns)])
        self.__all_widgets = widgets.VBox(children=[self.__progress, self.__buttons, self.__validation_show, self.__title_lbl, self.__map_classes])
 
    
    # create coco file format
    def __create_coco_format(self):
        self.__INFO = {
            "description": "Dataset",
            "url": "www.polimi.it",
            "version": "0.1.0",
            "year": 2020,
            "contributor": "Polimi",
            "date_created": datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        }
        self.__LICENSES = [
            {
                "id": 0,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

        self.__create_categories()
        self.__IMAGES = []
        self.__ANNOTATIONS = []
    
    # load already creted coco file
    def __load_coco_file(self):
        with open(self.__file_path, 'r') as f:
            dictionary_with_data = json.load(f)

        self.__INFO = dictionary_with_data['info']
        self.__LICENSES = dictionary_with_data['licenses']
        self.__CATEGORIES = dictionary_with_data['categories']
        self.__IMAGES = dictionary_with_data['images']
        self.__ANNOTATIONS = dictionary_with_data['annotations']


        assert set(self.__classes) == set(
            [c['name'] for c in self.__CATEGORIES]), "Classes in annotator and json must be equal"
    
    # get not used ids for annotations (deleted ann creates usable id)
    def __get_missing_ids(self, lst):
        ids = [el['id'] for el in lst]
        if len(ids) > 0:
            return sorted(set(range(min(ids), max(ids)+1)) - set(ids))
        return []
    
    # save current data to coco format
    def __save_coco_file(self):

        file_name = self.__get_image_name(self.__images[self.__current_img])
       
        # remove annotations and image from list (will be added again later)
        self.__ANNOTATIONS = list(filter(lambda x: x['image_id'] != self.__image_id, self.__ANNOTATIONS))
        self.__IMAGES = list(filter(lambda x: x['id'] != self.__image_id, self.__IMAGES))

        h, w, hh, ww, off_h, off_w = self.__img_coords[:]

        free_ann_ids = self.__get_missing_ids(self.__ANNOTATIONS)
        
        #print('Missing ids', free_ann_ids)

        for idx, ann in enumerate(self.__current_image_annotations):
            ann_cl = self.__current_class_for_ann[idx]
            coordinates = ann.bounds
            # rectangle opposite coordinates wrt geojson
            hs = [c[0] for c in coordinates]
            ws = [c[1] for c in coordinates]
            min_h, max_h = max(hs), min(hs)
            min_w, max_w = min(ws), max(ws)

            h_ratio = h / hh
            w_ratio = w / ww
            # map coords to img coords
            min_h = (min_h - off_h) * h_ratio
            max_h = (max_h - off_h) * h_ratio
            min_w = (min_w - off_w) * w_ratio
            max_w = (max_w - off_w) * w_ratio
            
            min_h = h - min_h
            max_h = h - max_h

            if len(free_ann_ids) > 0:
                ann_id = free_ann_ids.pop(0)
            else:
                ann_id = len(self.__ANNOTATIONS)

            annotation_info = {
                "id": ann_id,
                "image_id": self.__image_id,
                "category_id": ann_cl,
                "iscrowd": False,
                # "area": area.tolist(),
                "bbox": [min_w, min_h, max_w - min_w, max_h - min_h],
                # "segmentation": segmentation
            }
            self.__annotation_id = ann_id
            self.__ANNOTATIONS.append(annotation_info)
            #print('Saved id', ann_id)
        image_info = {
            "id": self.__image_id,
            "file_name": file_name,
            "width": w,
            "height": h,
            "date_captured": datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
            "license": 0,
            # "coco_url": coco_url,
            # "flickr_url": flickr_url
        }
        self.__IMAGES.append(image_info)

        dictionary_with_data = {'info': self.__INFO, 'licenses': self.__LICENSES, 'categories': self.__CATEGORIES,
                                'images': self.__IMAGES, 'annotations': self.__ANNOTATIONS}
        with open(self.__file_path, 'w') as f:
            json.dump(dictionary_with_data, f, indent=4)
    
    # get image name
    def __get_image_name(self, img_path):
        return os.path.basename(img_path)
    
    def __get_image_name_no_ext(self, img_path):
        return os.path.splitext(self.__get_image_name(img_path))[0]
    
    # get image id
    def __get_current_image_id(self):
        img_name = self.__get_image_name(self.__images[self.__current_img])
        
        for img in self.__IMAGES:
            if img['file_name'] == img_name:
                return img['id']
        if len(self.__IMAGES) > 0:
            return max(self.__IMAGES, key=lambda x: x['id'])['id'] + 1
        return 0
    
    # load existing annotations
    def __load_annotations(self):
        img_id = self.__get_current_image_id()
        # resize coords to fit in map
        h_i, w_i, hh, ww, offset_h, offset_w = self.__img_coords[:]
        h_ratio = hh / h_i
        w_ratio = ww / w_i

        for ann in self.__ANNOTATIONS:
            if ann['image_id'] == img_id:
                # create rectangle layer
                min_w, min_h, w, h = ann['bbox'][:]
                max_w = min_w + w
                max_h = min_h + h
                
                min_h = h_i - min_h
                max_h = h_i - max_h
                # coords to map coords
                min_h = min_h * h_ratio + offset_h
                max_h = max_h * h_ratio + offset_h
                min_w = min_w * w_ratio + offset_w
                max_w = max_w * w_ratio + offset_w

                rectangle = self.__create_rectangle(((min_h, min_w), (max_h, max_w)), default_class=ann['category_id'])
                rectangle.color = "green"
                rectangle.fill_color = "green"

                self.__map.add_layer(rectangle)
                self.__current_image_annotations.append(rectangle)
                self.__current_class_for_ann.append(ann['category_id'])
        #print('Current annotations', self.__current_class_for_ann)
    
    # create coco categories
    def __create_categories(self):
        self.__CATEGORIES = []
        for idx, cl in enumerate(self.__classes):
            proto = {
                'id': idx,
                'name': cl,
                'supercategory': ''
            }
            self.__CATEGORIES.append(proto)
    
    # create buttons for navigation
    def __create_btns(self):
        

        self.__next_button = widgets.Button(description=labels_str.str_btn_next)
        self.__reset_button = widgets.Button(description=labels_str.str_btn_reset)
        self.__previous_button = widgets.Button(description=labels_str.str_btn_prev)
        self.__delete_button = widgets.Button(description=labels_str.str_btn_delete_bbox, disabled=True)

        self.__next_button.on_click(self.__on_next_button_clicked)
        self.__reset_button.on_click(self.__on_reset_button_clicked)
        self.__previous_button.on_click(self.__on_previous_button_clicked)
        self.__delete_button.on_click(self.__on_delete)

        self.__buttons = widgets.HBox(children=[self.__previous_button, self.__next_button, 
                                                self.__delete_button, self.__reset_button])

    
    # create radio buttons with classes
    def __create_classes_btns(self):
        # TODO: CHANGE TO RADIO BUTTON
        self.__classes_btns = []
        for c in self.__classes:   
            layout = widgets.Layout(width='200px',height='25px')
            box = widgets.Checkbox(value=False,
                                   description=c,
                                   disabled=True,
                                   layout=layout,
                                   indent=False,
                                  )
            box.observe(self.__on_class_selected)
            self.__classes_btns.append(box)
        self.__classes_btns_interaction_disabled = False
    
    # reset all classes btns/checkbox/radio
    def __reset_classes_btns(self):
        # do not handle change value during reset
        self.__classes_btns_interaction_disabled = True
        for b in self.__classes_btns:
            b.value = False if self.__selected_class is None else b.description == self.__selected_class
            b.disabled = self.__selected_ann is None
        # enable handle change value for user interaction
        self.__classes_btns_interaction_disabled = False
    
    # interaction with classes btns/checkbox/radio
    def __on_class_selected(self, b):
        self.__clear_validation()
        if self.__classes_btns_interaction_disabled:
            return
        if self.__selected_ann is None:
            return
        if b['name'] == 'value':
            old_v = b['old']
            new_v = b['new']
            if new_v:
                self.__selected_class = b['owner'].description
                self.__current_class_for_ann[self.__selected_ann] = self.__classes.index(self.__selected_class)
            else:
                self.__selected_class = None
                self.__current_class_for_ann[self.__selected_ann] = None
            
            #print(old_v, new_v, self.__selected_class, self.__current_class_for_ann[self.__selected_ann])
                
            self.__reset_classes_btns()
        
    # click on rectangle layer
    def __handle_click(self, **kwargs):
        
        if kwargs.get('type') == 'click':
            self.__clear_validation()
            click_coords = kwargs.get('coordinates')
            clicked_ann = None
            # find clicked annotations (can be more than 1 if overlapping)
            clicked_size = None
            for idx, ann in enumerate(self.__current_image_annotations):
                coordinates = ann.bounds
                # rectangle opposite coordinates wrt geojson
                hs = [c[0] for c in coordinates]
                ws = [c[1] for c in coordinates]
                min_h, max_h = min(hs), max(hs)
                min_w, max_w = min(ws), max(ws)
                # don't break so if two rectangles are overlapping I take only the last drawed
                if min_h <= click_coords[0] <= max_h and min_w <= click_coords[1] <= max_w:  
                    curr_size = (max_h - min_h) * (max_w - min_w)
                    if clicked_size is None or curr_size < clicked_size:
                        clicked_size = curr_size
                        clicked_ann = idx
                        
            if clicked_ann is not None:

                # change color to green
                # +2 because layer 0 is map, layer 1 is overlay
                self.__selected_ann = clicked_ann
                self.__delete_button.disabled = False
                self.__reset_colors_bboxes()
                
                current_class = self.__current_class_for_ann[self.__selected_ann]
                self.__selected_class = None if current_class is None else self.__classes[current_class]
                self.__reset_classes_btns()

            # it should not enter here because click is handled only by annotations
            else:
                self.__selected_ann = None
                self.__selected_class = None
                self.__reset_colors_bboxes()
                self.__reset_classes_btns()
                self.__delete_button.disabled = True
                
    
    # reset bboxes to green or red colors
    def __reset_colors_bboxes(self):
        for i in range(len(self.__current_image_annotations)):
            # blue selected annotation
            if self.__selected_ann is not None and i == self.__selected_ann:
                self.__map.layers[i + 2].color = "blue"
                self.__map.layers[i + 2].fill_color = "blue"
            # red annotation without class
            elif self.__current_class_for_ann[i] is None:
                self.__map.layers[i + 2].color = "red"
                self.__map.layers[i + 2].fill_color = "red"
            # green annotation with class
            else:
                self.__map.layers[i + 2].color = "green"
                self.__map.layers[i + 2].fill_color = "green"
    
    # delete selected layer, class and geojson
    def __on_delete(self, b):
        if not self.__selected_ann is None:
            # +2 because layer 0 is map, layer 1 is overlay and rectangles start from index 2
            self.__map.remove_layer(self.__map.layers[self.__selected_ann + 2])
            self.__current_image_annotations.pop(self.__selected_ann)
            self.__current_class_for_ann.pop(self.__selected_ann)
            # print('deleted')
            self.__selected_ann = None
            self.__selected_class = None
            self.__reset_colors_bboxes()
            self.__reset_classes_btns()
            self.__delete_button.disabled = True

        

    # create annotation rectangle
    def __create_rectangle(self, bounds, default_class=None):
        rectangle = Rectangle(bounds=bounds, color="red", fill_color="red")
        rectangle.on_click(self.__handle_click)
        mid_h = bounds[0][0] + (bounds[1][0] - bounds[0][0]) / 2
        mid_w = bounds[0][1] + (bounds[1][1] - bounds[0][1]) / 2
        #rectangle.popup = self.__create_popup((mid_h, mid_w), default_class)
        return rectangle
    
    # create and handle draw control
    def __create_draw_control(self):
        dc = DrawControl(rectangle={'shapeOptions': {'color': '#0000FF'}}, circle={}, circlemarker={}, polyline={},
                         marker={}, polygon={})
        dc.edit = False
        dc.remove = False

        # handle drawing and deletion of annotations and corresponding classes
        def handle_draw(target, action, geo_json):
            # print(target)
            # print(action)
            # print(geo_json)
            if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
                coordinates = geo_json['geometry']['coordinates'][0]
                #print(coordinates)
                #print(self.__map)
                #print(coordinates)
                hs = [c[1] for c in coordinates]
                ws = [c[0] for c in coordinates]
                min_h, max_h = min(hs), max(hs)
                min_w, max_w = min(ws), max(ws)
                
                # coordinates only inside image
                hh, ww, offset_h, offset_w = self.__img_coords[2:]
                max_h = max(0, min(hh + offset_h, max_h))
                max_w = max(0, min(ww + offset_w, max_w))
                min_h = max(offset_h, min(hh + offset_h, min_h))
                min_w = max(offset_w, min(ww + offset_w, min_w))
                
                
                # remove draw
                dc.clear()

                if max_h - min_h < 1 or max_w - min_w < 1:
                    print(labels_str.warn_skip_wrong )
                    return

                # print(min_h, max_h, min_w, max_w)
                # create rectangle layer and remove drawn geojson
                rectangle = self.__create_rectangle(((min_h, min_w), (max_h, max_w)))
                self.__current_image_annotations.append(rectangle)
                self.__current_class_for_ann.append(None)
                self.__map.add_layer(rectangle)
                # automatically select last annotation
                self.__selected_ann = len(self.__current_image_annotations)-1
                self.__reset_colors_bboxes()
                self.__selected_class = None
                self.__reset_classes_btns()
                self.__delete_button.disabled = False
                # print('Adding ann at index:',len(self.current_image_annotations)-1,
                # ' with class', self.current_class_for_ann[-1])

        dc.on_draw(handle_draw)
        self.__map.add_control(dc)
    
    # load image and get map overlay
    def __get_img_overlay(self, img_path):
        # We need to create an ImageOverlay for each image to show,
        # and set the appropriate bounds based  on the image size
              
        if not os.path.exists(img_path):
            print(labels_str.warn_img_path_not_exits +  img_path)
            

        im = cv2.imread(img_path)
        h, w, _ = im.shape

        max_v = 100
        
        offset_h = -25
        offset_w = -25
        hh = max_v - offset_h*2
        ww = max_v - offset_w*2
        
        if h > w:
            ww = int(w * hh / h)
            offset_w = (max_v - ww) / 2
        elif w > h:
            hh = int(h * ww / w)
            offset_h = (max_v - hh) / 2

        img_ov = ImageOverlay(url=img_path, bounds=((offset_h, offset_w), (hh + offset_h, ww+offset_w)))
        return img_ov, h, w, hh, ww, offset_h, offset_w
    
    
    # create and set map
    def __create_map(self):
        # Create the "map" that will be used to display the image
        # the crs simple, indicates that we will use pixels to locate objects inside the map
        self.__map = Map(center=(50, 50), zoom=2, crs=projections.Simple, dragging=False, 
                         zoom_control=False, double_click_zoom=False,
                         layers=[LocalTileLayer(path='white.png')], layout=dict(width='600px', height='600px'))

        self.__create_draw_control()
    
    # remove all annotations from map
    def __clear_map(self, keep_img_overlay=False):
        starting_layer = 2 if keep_img_overlay else 1
        for l in self.__map.layers[starting_layer:]:
            self.__map.remove_layer(l)
        self.__current_image_annotations = []
        self.__current_class_for_ann = []
        self.__selected_class = None
        self.__selected_ann = None
        self.__delete_button.disabled = True
    
    # disable or enable buttons
    def __toggle_interaction_buttons(self, disabled):
        self.__next_button.disabled = disabled
        self.__previous_button.disabled = disabled
        self.__reset_button.disabled = disabled
    
    # enable or disable buttons for first and last images
    def __change_buttons_status(self):
        self.__next_button.disabled = self.__current_img >= len(self.__images) - 1
        self.__previous_button.disabled = self.__current_img <= 0
        
    def __clear_workspace(self):
        self.__toggle_interaction_buttons(disabled=True)
        self.__save_coco_file()
        self.__clear_map()
        self.__reset_classes_btns()
        self.__clear_validation()
    
    def __clear_validation(self):
        self.__validation_show.value = ""
        
    # next button clicked
    def __on_next_button_clicked(self, b):
        if None in self.__current_class_for_ann:
            self.__validation_show.value = labels_str.warn_select_class
            return
        
        self.__clear_workspace()
        self.__show_image(1)
        
    
    # previous button clicked
    def __on_previous_button_clicked(self, b):
        if None in self.__current_class_for_ann:
            return
        
        self.__clear_workspace()
        self.__show_image(-1)
        
   # reset button clicked
    def __on_reset_button_clicked(self, b):
        self.__clear_map(keep_img_overlay=True)
        self.__reset_classes_btns()
    
    # show current image
    def __show_image(self, delta):
        # update progress bar
        self.__current_img += delta
        self.__progress.value = self.__current_img + 1
        self.__progress.description = '{}/{}'.format(self.__current_img + 1, len(self.__images))
        self.__toggle_interaction_buttons(disabled=False)
        # change buttons
        self.__change_buttons_status()
        # update map
        img_ov, h, w, hh, ww, off_h, off_w = self.__get_img_overlay(self.__images[self.__current_img])
        self.__img_coords = (h, w, hh, ww, off_h, off_w)
        self.__map.add_layer(img_ov)
        self.__image_id = self.__get_current_image_id()
        # update title
        file_name = self.__get_image_name_no_ext(self.__images[self.__current_img])
        self.__title_lbl.value = file_name
        if self.__images_classes is not None:
            pass
            # do something with additional infos
        # load current annotations
        self.__load_annotations()

    def start_annotation(self):
        display(self.__all_widgets)
        self.__show_image(1)