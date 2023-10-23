from modules import script_callbacks, postprocessing
import gradio as gr
from PIL import Image
import numpy as np

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

def on_ui_tab_called():
    with gr.Blocks() as transparent_interface:
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("PNG2APNG"):
                    image_upload_input = gr.Image(label="Upload Image", source= "upload",type="pil")
                    threshold_input = gr.Slider(minimum=0, maximum=255, value=100, label="Threshold")
                    upscale_input = gr.Slider(minimum=1, maximum=8, value=1, label="Upscale, 1 to disable")
                    button = gr.Button(label="Convert")
                    image_upload_output = gr.Image(label="Output Image",type="numpy")
                    def convert_image(image:Image.Image, threshold:float, upscale_scale:float)->np.ndarray:
                        """
                        Converts the image to apng
                        The black color (with some threshold) will remain, others will be transparent
                        """
                        color_threshold = threshold
                        print("Threshold:", color_threshold)
                        print("Upscale:", upscale_scale)
                        # upscale the image
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
                    button.click(convert_image, inputs=[image_upload_input, threshold_input, upscale_input], outputs=[image_upload_output])
    return (transparent_interface, "PNG2APNG", "script_png2apng_interface"),

script_callbacks.on_ui_tabs(on_ui_tab_called)