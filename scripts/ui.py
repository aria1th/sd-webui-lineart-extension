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
                    threshold_input = gr.Slider(minimum=0, maximum=255, value=20, label="Threshold")
                    button = gr.Button(label="Convert")
                    image_upload_output = gr.Image(label="Output Image",type="pil")
                    
                    def convert_image(image:Image.Image, threshold:float)->Image.Image:
                        """
                        Converts the image to apng
                        The black color (with some threshold) will remain, others will be transparent
                        """
                        color_threshold = threshold
                        # first convert to RGB
                        image = image.convert("RGB")
                        # get the pixels that has black or color that is close to black
                        arr = np.array(image) # convert to numpy array, channel 3
                        # get the black pixels
                        black_pixels = np.where(np.all(arr <= color_threshold, axis=-1))
                        # create new apng image
                        apng_shape = (image.height, image.width, 4)
                        new_image = np.zeros(apng_shape, dtype=np.uint8)
                        # put the white pixels
                        new_image[:,:,:4] = (255,255,255,255)
                        # put the black pixels
                        new_image[black_pixels] = [0,0,0,255]
                        # convert to PIL image
                        new_image = Image.fromarray(new_image)
                        return new_image
                    button.click(convert_image, inputs=[image_upload_input, threshold_input], outputs=[image_upload_output])
    return (transparent_interface, "PNG2APNG", "script_png2apng_interface"),

script_callbacks.on_ui_tabs(on_ui_tab_called)