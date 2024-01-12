import gradio as gr


import numpy as np
from PIL import Image, ImageEnhance
from scripts.lineart_functions import upscale_image

def white_to_transparent_hsv_pil(img, sat_threshold=30, val_threshold=220):
    img = img.convert('HSV')
    data = np.array(img)
    hue, sat, val = data.T
    white_areas = (sat < sat_threshold) & (val > val_threshold)
    img = img.convert('RGBA')
    data = np.array(img)
    data[..., -1][white_areas.T] = 0
    img_with_transparency = Image.fromarray(data)
    return img_with_transparency

def brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(factor)

    return brightened_image

def processing_v2(image, scale:float = 4, brightness_value:float = 0.3):
    image = upscale_image(image, scale)
    image = white_to_transparent_hsv_pil(image)
    image = brightness(image, brightness_value)
    return image
