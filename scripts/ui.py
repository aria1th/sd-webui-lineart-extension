import gradio as gr
from PIL import Image
import numpy as np
import cv2
from PIL import ImageFilter, ImageOps
from scripts.webuiapi import WebUIApi

def upscale_image(image:Image.Image, scale:int)->Image.Image:
    """
    Upscales the image
    """
    if scale == 1:
        return image
    # if image is numpy array, convert to PIL image
    if isinstance(image, np.ndarray):
        # RGBA to RGB, convert transparent pixels to white
        if image.shape[2] == 4:
            #print(f"Upscaling RGBA image : {image.shape}")
            white_image = Image.new("RGB", image.shape[:2][::-1], (255, 255, 255)) # numpy array is height, width
            white_image.paste(Image.fromarray(image), mask=Image.fromarray(image).split()[3])
            image = white_image
        else:
            image = Image.fromarray(image)
    else:
        # image, handle RGBA
        if image.mode == "RGBA":
            # convert transparent pixels to white
            white_image = Image.new("RGB", image.size, (255, 255, 255))
            white_image.paste(image, mask=image.split()[3])
            image = white_image
        else:
            image = image.convert("RGB")
    #print(image.size)
    api_instance = WebUIApi(
        port=9050 #
    )
    upscale_result = api_instance.extra_single_image(
        image=image,
        upscaler_1 = "R-ESRGAN 4x+ Anime6B",
        upscaling_resize_h= image.height * scale,
        upscaling_resize_w= image.width * scale,
    )
    images = upscale_result[0]
    return images[0]

def gaussian_and_re_threshold(image:Image.Image, threshold:int, blur:int)->Image.Image:
    """
    Applies small gaussian blur and re-threshold the image
    image : RGBA image
    """
    white_image = Image.new("RGB", image.size, (255, 255, 255))
    white_image.paste(image, mask=image.split()[3])
    # convert to grayscale
    grayscale_image = white_image.convert("L")
    # convert to numpy array
    array = np.array(grayscale_image)
    # blur the image
    if blur > 1:
        blurred_image = cv2.GaussianBlur(array, (blur,blur), 0)
    else:
        blurred_image = array # no blur
    # use standard threshold to get black pixels
    _, threshold_result = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
    # create new apng image
    apng_shape = (image.height, image.width, 4)
    new_image = np.zeros(apng_shape, dtype=np.uint8)
    # put the black pixels  
    new_image[threshold_result == 0] = [0,0,0,255]
    new_image = Image.fromarray(new_image)
    assert new_image.size == image.size, f"New image size is different from original image size : {new_image.size} vs {image.size}"
    return new_image

def small_points_remover(image:Image.Image, threshold:int) -> Image.Image:
    # convert RGBA to cv2.COLOR_BGR2GRAY
    white_image = Image.new("RGB", image.size, (255, 255, 255))
    white_image.paste(image, mask=image.split()[3])
    grayscale_image = cv2.cvtColor(np.array(white_image), cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(grayscale_image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # contour의 가로외 세로중에 큰 값이 100이상인 것만 추출
    contours = [i for i in contours if cv2.boundingRect(i)[2] > threshold or cv2.boundingRect(i)[3] > threshold]
    
    mask = np.zeros_like(white_image)
    cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    
    # convert mask to RGBA
    apng_shape = (image.height, image.width, 4)
    new_image = np.zeros(apng_shape, dtype=np.uint8)
    # put the mask's black pixels to new image
    new_image[mask[:,:,0] == 0] = [0,0,0,255]
    new_image = Image.fromarray(new_image)
    assert new_image.size == image.size, f"New image size is different from original image size : {new_image.size} vs {image.size}"
    return new_image


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
                        button = gr.Button(label="Convert")
                    with gr.Row():
                        image_upload_output = gr.Image(label="Output Image",type="numpy")
                    def convert_image(image:Image.Image, threshold:float, blur:float, remove:float, upscale_scale:float, adaptive:bool, upscale_order:bool)->np.ndarray:
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
                    
                    button.click(convert_image, inputs=[image_upload_input, threshold_input, threshold_blur, threshold_remove, upscale_input, adaptive_checkbox, upscale_order_checkbox], outputs=[image_upload_output])
    return (transparent_interface, "PNG2APNG", "script_png2apng_interface"),

try:
    from modules import script_callbacks, postprocessing
    script_callbacks.on_ui_tabs(on_ui_tab_called)
except (ImportError, ModuleNotFoundError):
    # context not in webui, run as separate script
    if __name__ == "__main__":
        interface, _, _ = on_ui_tab_called()[0]
        interface.launch()