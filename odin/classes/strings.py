str_btn_prev = "Previous"
str_btn_next = "Next"
str_btn_download = "Download"
str_btn_reset = "Reset Annotation"
str_btn_delete_bbox = "Delete Bbox"

srt_validation_not_ok = "<b style=\"color:green\">NO ACTION REQUIRED</b>"
srt_validation_ok = "<b style=\"color:RED\">ACTION REQUIRED</b>"

info_new_ds = "New dataset with meta_annotations"
info_missing = "Missing image"
info_no_more_images = "No available images"

info_total = "Total"
info_class = "Class"
info_class_name = "Class name"
info_ann_images = "Annotated images"
info_ann_objects = "Annotated objects"
info_positions = "Positions:"

info_completed = "Completed images:"
info_incomplete = "Incomplete images:"
info_completed_obj = "Completed objects:"
info_incomplete_obj = "Incomplete objects:"
info_ds_output = "Generated dataset will be saved at: "

info_loading_properties = "Loading properties..."
info_done = "Done!"

warn_select_class = "<b style=\"color:RED\">You must assign a class to all bbox before continuing</b>"
warn_skip_wrong = "Skipping wrong annotation"
warn_img_path_not_exits = "Image path does not exists "

warn_task_not_supported = "Task type not supported"
warn_no_images = "No images provided"
warn_little_classes = "At least one class must be provided"
warn_binary_only_two = "Binary datasets contain only 2 classes"
warn_display_function_needed = "Non image data requires the definition of the custom display function"


warn_no_images_criteria = "No images meet the specified criteria"
warn_incorrect_class = "Class not present in dataset"
warn_incorrect_property = "Meta-Annotation not present in dataset"

warn_no_proposals = "No proposals available. Please make sure to load the proposals to the dataset"
warn_no_properties = "No properties available. Please make sure to load the properties to the dataset"

err_properties_file = "Error loading properties file"
err_categories_file = "Error loading categories file"

err_categories_id_dataset = "categories: mandatory field 'id' not found"
err_categories_id_dataset_few = "categories: mandatory field 'id' not found in some categories"
err_categories_name_dataset = "categories: mandatory field 'name' not found"
err_categories_name_dataset_few = "categories: mandatory field 'name' not found in some categories"
err_images_id_dataset = "images: mandatory field 'id' not found"
err_images_id_dataset_few = "images: mandatory field 'id' not found in some images"
err_images_filename_dataset = "images: mandatory field 'file_name' not found. Please make sure to specify this field " \
                              "if match_on_filename=True"
err_images_filename_dataset_few = "images: mandatory field 'file_name' not found in some images"
err_annotations_image_id_dataset = "annotations: mandatory field 'image_id' not found"
err_annotations_image_id_dataset_few = "annotations: mandatory field 'image_id' not found in some images"
err_annotations_category_id_dataset = "annotations: mandatory field 'category_id' not found"
err_annotations_category_id_dataset_few = "annotations: mandatory field 'category_id' not found in some annotations"
err_annotations_segmentation_dataset = "annotations: mandatory field 'segmentation' not found. Please make sure to " \
                                       "specify this field for TaskType.INSTANCE_SEGMENTATION"
err_annotations_segmentation_dataset_few = "annotations: mandatory field 'segmentation' not found in some annotations"
err_annotations_bbox_dataset = "annotations: mandatory field 'bbox' not found. Please make sure to specify this " \
                               "field for TaskType.OBJECT_DETECTION"
err_annotations_bbox_dataset_few = "annotations: mandatory field 'bbox' not found in some annotations"
err_observations_id_dataset = "observations: mandatory field 'id' not found"
err_observations_id_dataset_few = "observations: mandatory field 'id' not found in some observations"
err_observations_filename_dataset = "observations: mandatory field 'file_name' not found. Please make sure to specify " \
                                    "this field if match_on_filename=True"
err_observations_filename_dataset_few = "observations: mandatory field 'file_name' not found in some observations"
err_observations_categories_dataset = "observations: mandatory field 'categories' not found. Please make sure to " \
                                      "specify this field for TaskType.CLASSIFICATION_MULTI_LABEL"
err_observations_categories_dataset_few = "observations: mandatory field 'categories' not found in some observations"
err_observations_category_dataset = "observations: mandatory field 'category' not found. Please make sure to specify " \
                                    "this field for TaskType.CLASSIFICATION_SINGLE_LABEL and " \
                                    "TaskType.CLASSIFICATION_BINARY"
err_observations_category_dataset_few = "observations: mandatory field 'category' not found in some observations"


err_analyzer_invalid_curve = "Invalid '{}' curve"

err_type = "Invalid '{}' type"
err_value = "Invalid '{}' value. Possible values: {}"
err_property_not_loaded = "Property '{}' not loaded"

# Time series strings
err_ts_metric = "Not supported metric: {}"
err_ts_properties_not_loaded = "Properties file has not been loaded. Load it before trying to use it."
err_ts_properties_json = "Error loading the properties json. The file path you specified is: {}"
err_ts_property_invalid = "The specified property [{}] is invalid."
err_ts_property_values_invalid = "At least one property values you specified is invalid."
err_ts_property_file_format_invalid = "The dataset json file format is invalid for properties."
err_ts_only_interval_interval = "This analysis is only available for INTERVAL-INTERVAL matching strategies"
