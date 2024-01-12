import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
try:
    from modules import postprocessing
    upscale_enabled = True
except:
    upscale_enabled = False

def saturate(image, scale):
    """
    Saturation filter
    """
    if scale == 1:
        return image
    enhancer = ImageEnhance.Color(image)
    saturated_image = enhancer.enhance(scale)
    return saturated_image

def upscale_image(image:Image.Image, scale:int)->Image.Image:
    """
    Upscales the image
    """
    if scale == 1 or not upscale_enabled:
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
        upscaling_resize_w= image.width * scale,
        upscaling_resize_h= image.height * scale,
        upscaling_crop= True,
        extras_upscaler_2= 'None',
        extras_upscaler_2_visibility= 0,
        upscale_first= False,
    )
    images = result[0]
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