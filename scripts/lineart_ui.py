import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
from scripts.extra_tab_whiteremove import processing_v2_binary
from scripts.lineart_functions import saturate, upscale_image, gaussian_and_re_threshold, small_points_remover

def on_ui_tab_called():
    with gr.Blocks() as transparent_interface:
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("Invert"):
                    with gr.Row():
                        image_invert_input = gr.Image(label="Upload Image", source= "upload",type="pil")
                        button = gr.Button(label="Invert")
                    with gr.Row():
                        image_invert_output = gr.Image(label="Output Image",type="numpy")
                    def invert_image(image:Image.Image)->np.ndarray:
                        # imageops.invert
                        # handle RGBA
                        if image.mode == "RGBA":
                            # convert transparent pixels to white
                            white_image = Image.new("RGB", image.size, (255, 255, 255))
                            white_image.paste(image, mask=image.split()[3])
                            image = white_image
                        return ImageOps.invert(image)
                    button.click(invert_image, inputs=[image_invert_input], outputs=[image_invert_output])
                with gr.TabItem("PNG2APNG"):
                    with gr.Row():
                        image_upload_input = gr.Image(label="Upload Image", source= "upload",type="pil")
                        with gr.Row():
                            threshold_input = gr.Slider(minimum=0, maximum=255, value=100, label="Color Threshold for black")
                            threshold_blur = gr.Slider(minimum=1, maximum=13, value=1, step = 2, label="Blur strength")
                            threshold_remove = gr.Slider(minimum=0, maximum=255, value=50, label="Remove small points threshold")
                            upscale_input = gr.Slider(minimum=1, maximum=8, value=1, label="Upscale, 1 to disable")
                            adaptive_checkbox = gr.Checkbox(label="Adaptive Threshold", value=False)
                            upscale_order_checkbox = gr.Checkbox(label="Upscale first", value=False)
                            saturation_input = gr.Slider(minimum=0, maximum=10, value=5, label="Saturation")
                        button = gr.Button(label="Convert")
                    with gr.Row():
                        image_upload_output = gr.Image(label="Output Image",type="numpy")
                    def convert_image(image:Image.Image, threshold:float, blur:float, 
                                      remove:float, upscale_scale:float, adaptive:bool,
                                      upscale_order:bool, saturate_scale: int)->np.ndarray:
                        """
                        Converts the image to apng
                        The black color (with some threshold) will remain, others will be transparent
                        
                        @param upscale_order : if True, upscale the image first, then calculate line art, else revert
                        """
                        color_threshold = threshold
                        remove_threshold = remove
                        threshold_blur = blur
                        # upscale the image
                        # convert to RGB
                        #print(f"Base image size : {image.size}")
                        if upscale_order:
                            image = upscale_image(image, upscale_scale)
                            #print(f"Upscaled image size : {image.size}")
                        # first convert to RGB
                        # warn : APNG transparent channels should be converted as white
                        # saturate the image
                        image = saturate(image, saturate_scale)
                        if image.mode == "RGBA":
                            # convert transparent pixels to white
                            white_image = Image.new("RGB", image.size, (255, 255, 255))
                            white_image.paste(image, mask=image.split()[3])
                            image = white_image
                        else:
                            image = image.convert("RGB")
                        #print(f"Converted image size : {image.size}")
                        # get the pixels that has black or color that is close to black
                        # Using HSV color space
                        # convert to HSV
                        if not adaptive:
                            hsv_image = image.convert("HSV")
                            # assert size is same
                            assert hsv_image.size == image.size, f"HSV image size is different from RGB image size : {hsv_image.size} vs {image.size}"
                            # get the pixels that has black or color that is close to black, we can use brightness
                            array = np.array(hsv_image)
                            #print(f"HSV image size : {hsv_image.size}")
                            #print(f"HSV image array shape : {array.shape}") # width and height order is reversed
                            # get the brightness
                            brightness = array[:,:,2]
                            # brightness should be less than the threshold
                            black_pixels = brightness <= color_threshold
                            # create new apng image with white background
                            apng_shape = (image.height, image.width, 4)
                            new_image = np.zeros(apng_shape, dtype=np.uint8)
                            # put the black pixels
                            new_image[black_pixels] = [0,0,0,255]
                            #print(f"Non-adaptive image size : {new_image.shape}")
                        else:
                            # use adaptive gaussian threshold
                            # convert to grayscale
                            grayscale_image = image.convert("L")
                            # convert to numpy array
                            array = np.array(grayscale_image)
                            # using cv2 adaptive gaussian threshold
                            threshol_result = cv2.adaptiveThreshold(array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
                            # create new apng image
                            apng_shape = (image.height, image.width, 4)
                            new_image = np.zeros(apng_shape, dtype=np.uint8)
                            # put the black pixels
                            new_image[threshol_result == 0] = [0,0,0,255]
                            new_image = Image.fromarray(new_image)
                            new_image = gaussian_and_re_threshold(new_image, color_threshold, threshold_blur)
                            new_image = small_points_remover(new_image, remove_threshold)
                            #print(f"Adaptive image size : {new_image.size}")
                        # upscale the image
                        if not upscale_order:
                            #print(f"pre-Upscaled image size : {new_image.shape}")
                            new_image = upscale_image(new_image, upscale_scale) #RGB Image
                            #print(f"Upscaled image size : {new_image.size}")
                            if upscale_scale > 1:
                                # RGB image to RGBA
                                # we will set white pixels to transparent
                                # convert to numpy array
                                array = np.array(new_image) # 3D array
                                # get the white pixels
                                white_pixels = np.all(array >= 250, axis=-1)
                                # put the transparent pixels
                                rgba_array = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
                                # put all the pixels first
                                rgba_array[:,:,:3] = array
                                # set all alpha to 255
                                rgba_array[:,:,3] = 255
                                # put the transparent pixels
                                rgba_array[white_pixels] = [0,0,0,0]
                                # convert to image
                                new_image = Image.fromarray(rgba_array)
                            
                        return new_image # return the new image
                    
                    button.click(convert_image, inputs=[image_upload_input, threshold_input, threshold_blur, threshold_remove, upscale_input, adaptive_checkbox, upscale_order_checkbox, saturation_input], outputs=[image_upload_output])
                with gr.TabItem("PNG2APNGv2"):
                    image_input = gr.Image(label="Upload Image", source= "upload",type="pil")
                    upscale_scaler = gr.Slider(minimum=1, maximum=8, value=4, step=0.5, label="Upscale, 1 to disable")
                    brightness_scale = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.05, label="Brightness")
                    button = gr.Button(label="Convert")
                    image_output_file = gr.File(label="Output Image (binary)", type="file")
                    
                    button.click(processing_v2_binary, inputs=[image_input, upscale_scaler, brightness_scale], outputs=[image_output_file])

    return (transparent_interface, "PNG2APNG", "script_png2apng_interface"),

try:
    from modules import script_callbacks, postprocessing
    script_callbacks.on_ui_tabs(on_ui_tab_called)
except (ImportError, ModuleNotFoundError):
    # context not in webui, run as separate script
    if __name__ == "__main__":
        interface, _, _ = on_ui_tab_called()[0]
        interface.launch()