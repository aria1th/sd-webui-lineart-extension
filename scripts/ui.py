from modules import script_callbacks
import gradio as gr
from PIL import Image
import numpy as np

def on_ui_tab_called():
    with gr.Blocks() as transparent_interface:
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("PNG2APNG"):
                    image_upload_input = gr.Image(label="Upload Image", source="upload",type="pil")
                    threshold_input = gr.Slider(minimum=0, maximum=255, value=100, label="Threshold")
                    button = gr.Button(label="Convert")
                    image_upload_output = gr.Image(label="Output Image",type="numpy")
                    
                    def convert_image(image:Image.Image, threshold:float)->np.ndarray:
                        """
                        Converts the image to apng
                        The black color (with some threshold) will remain, others will be transparent
                        """
                        color_threshold = threshold
                        print("Threshold:", color_threshold)
                        # first convert to RGB
                        image = image.convert("RGB")
                        # get the pixels that has black or color that is close to black
                        # Using HSV color space
                        # convert to HSV
                        hsv_image = image.convert("HSV")
                        # get the pixels that has black or color that is close to black, we can use brightness
                        array = np.array(hsv_image)
                        # get the brightness
                        brightness = array[:,:,2]
                        # brightness should be less than the threshold
                        black_pixels = brightness <= color_threshold
                        # create new apng image
                        apng_shape = (image.height, image.width, 4)
                        new_image = np.zeros(apng_shape, dtype=np.uint8)
                        # put the black pixels
                        new_image[black_pixels] = [0,0,0,255]
                        return new_image # return the new image
                    button.click(convert_image, inputs=[image_upload_input, threshold_input], outputs=[image_upload_output])
    return (transparent_interface, "PNG2APNG", "script_png2apng_interface"),

script_callbacks.on_ui_tabs(on_ui_tab_called)