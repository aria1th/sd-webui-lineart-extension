import gradio as gr
from PIL import Image
import numpy as np
import cv2
from PIL import ImageFilter

def upscale_image(image:Image.Image, scale:int)->Image.Image:
    """
    Upscales the image
    """
    if scale == 1:
        return image
    extra_upscale_func = postprocessing.run_extras
    result = extra_upscale_func(
        extras_mode=0,
        resize_mode=0,
        upscaling_resize = scale,
        extras_upscaler_1="R-ESRGAN 4x+ Anime6B",
        image=image,
        image_folder= "",
        input_dir= "",
        output_dir= "",
        show_extras_results= 0,
        gfpgan_visibility= 0,
        codeformer_visibility= 0,
        codeformer_weight= 0,
        upscaling_resize_w= 512,
        upscaling_resize_h= 512,
        upscaling_crop= True,
        extras_upscaler_2= 'None',
        extras_upscaler_2_visibility= 0,
        upscale_first= False,
    )
    images = result[0]
    return images[0]

def gaussian_and_re_threshold(image:Image.Image, threshold:int)->Image.Image:
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
    blurred_image = cv2.GaussianBlur(array, (9,9), 0)
    # use standard threshold to get black pixels
    _, threshold_result = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
    # create new apng image
    apng_shape = (image.height, image.width, 4)
    new_image = np.zeros(apng_shape, dtype=np.uint8)
    # put the black pixels
    new_image[threshold_result == 0] = [0,0,0,255]
    new_image = Image.fromarray(new_image)
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
    return new_image


def on_ui_tab_called():
    with gr.Blocks() as transparent_interface:
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("PNG2APNG"):
                    with gr.Row():
                        image_upload_input = gr.Image(label="Upload Image", source= "upload",type="pil")
                        threshold_input = gr.Slider(minimum=0, maximum=255, value=100, label="Threshold_black")
                        threshold_remove = gr.Slider(minimum=0, maximum=255, value=50, label="Threshold_remove")
                        upscale_input = gr.Slider(minimum=1, maximum=8, value=1, label="Upscale, 1 to disable")
                        adaptive_checkbox = gr.Checkbox(label="Adaptive Threshold", value=False)
                        button = gr.Button(label="Convert")
                    with gr.Row():
                        image_upload_output = gr.Image(label="Output Image",type="numpy")
                    def convert_image(image:Image.Image, threshold:float, remove:float, upscale_scale:float, adaptive:bool)->np.ndarray:
                        """
                        Converts the image to apng
                        The black color (with some threshold) will remain, others will be transparent
                        """
                        color_threshold = threshold
                        remove_threshold = remove
                        # upscale the image
                        # convert to RGB
                        image = upscale_image(image, upscale_scale)
                        # first convert to RGB
                        # warn : APNG transparent channels should be converted as white
                        if image.mode == "RGBA":
                            # convert transparent pixels to white
                            white_image = Image.new("RGB", image.size, (255, 255, 255))
                            white_image.paste(image, mask=image.split()[3])
                            image = white_image
                        else:
                            image = image.convert("RGB")
                        # get the pixels that has black or color that is close to black
                        # Using HSV color space
                        # convert to HSV
                        if not adaptive:
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
                        else:
                            # use adaptive gaussian threshold
                            # convert to grayscale
                            grayscale_image = image.convert("L")
                            # convert to numpy array
                            array = np.array(grayscale_image)
                            # using cv2 adaptive gaussian threshold
                            threshol_result = cv2.adaptiveThreshold(array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)
                            # create new apng image
                            apng_shape = (image.height, image.width, 4)
                            new_image = np.zeros(apng_shape, dtype=np.uint8)
                            # put the black pixels
                            new_image[threshol_result == 0] = [0,0,0,255]
                            new_image = Image.fromarray(new_image)
                            new_image = gaussian_and_re_threshold(new_image, color_threshold)
                            new_image = small_points_remover(new_image, remove_threshold)
                        return new_image # return the new image
                    
                    button.click(convert_image, inputs=[image_upload_input, threshold_input, threshold_remove, upscale_input, adaptive_checkbox], outputs=[image_upload_output])
    return (transparent_interface, "PNG2APNG", "script_png2apng_interface"),

try:
    from modules import script_callbacks, postprocessing
    script_callbacks.on_ui_tabs(on_ui_tab_called)
except (ImportError, ModuleNotFoundError):
    # context not in webui, run as separate script
    if __name__ == "__main__":
        interface, _, _ = on_ui_tab_called()[0]
        interface.launch()