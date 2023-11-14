from collections import defaultdict
import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any
import uuid
import threading

        

class Upscaler(str, Enum):
    none = "None"
    Lanczos = "Lanczos"
    Nearest = "Nearest"
    LDSR = "LDSR"
    BSRGAN = "BSRGAN"
    ESRGAN_4x = "ESRGAN_4x"
    R_ESRGAN_General_4xV3 = "R-ESRGAN General 4xV3"
    ScuNET_GAN = "ScuNET GAN"
    ScuNET_PSNR = "ScuNET PSNR"
    SwinIR_4x = "SwinIR 4x"


class HiResUpscaler(str, Enum):
    none = "None"
    Latent = "Latent"
    LatentAntialiased = "Latent (antialiased)"
    LatentBicubic = "Latent (bicubic)"
    LatentBicubicAntialiased = "Latent (bicubic antialiased)"
    LatentNearest = "Latent (nearist)"
    LatentNearestExact = "Latent (nearist-exact)"
    Lanczos = "Lanczos"
    Nearest = "Nearest"
    ESRGAN_4x = "ESRGAN_4x"
    LDSR = "LDSR"
    ScuNET_GAN = "ScuNET GAN"
    ScuNET_PSNR = "ScuNET PSNR"
    SwinIR_4x = "SwinIR 4x"


@dataclass
class WebUIApiResult:
    images: list
    parameters: dict
    info: dict

    @property
    def image(self):
        return self.images[0]
    
    def get_image(self):
        return self.images[0]
    
class QueuedTaskResult:
    task_id: str
    task_address: str # address to get task status
    image: str = ""# base64 encoded image
    images: List[str] = []
    terminated: bool = False
    cached_image: Image.Image = None
    cached_images: List[Image.Image] = []
    infotexts:list[str] = [] # list of infotexts for each image
    
    def __init__(self, task_id: str, task_address: str, session: requests.Session = None):
        self.task_id = task_id
        self.task_address = task_address
        self.expect_rate_limit = True
        self.session = session if session else requests.Session()

    def get_image(self):
        self.check_finished()
        if not self.terminated:
            return None
        if self.cached_image is None:
            self.cached_image = Image.open(io.BytesIO(base64.b64decode(self.image.split(',')[-1])))
        return self.cached_image
    
    def get_images(self):
        self.check_finished()
        if not self.terminated:
            return None
        if not self.cached_images: # if empty
            self.cached_images = [Image.open(io.BytesIO(base64.b64decode(img.split(',')[-1]))) for img in self.images]
        return self.cached_images
    
    def is_finished(self):
        """
        Returns True if task is finished
        Returns False if task is not finished
        Throws RuntimeError if task is failed
        Throws ValueError if task is not found
        """
        self.check_finished()
        return self.terminated
    
    def check_finished(self, check_delay:int = 5):
        """
        check if task is finished
        Throws RuntimeError if task is failed
        Throws ValueError if task is not found
        """
        if not self.terminated:
            # self.task_address is base address, /agent-scheduler should be added
            # check /agent-scheduler/v1/queue
            # it should return {"current_task_id" : str, "pending_tasks" : [{"api_task_id" : str}]}
            # if self.task_id is found in any of pending tasks or current_task_id, then it is not finished
            # else, find /agent-scheduler/v1/results/{task_id}
            response = self.session.get(self.task_address + "/agent-scheduler/v1/queue") 
            try:
                req_json = response.json()
            except json.JSONDecodeError as exc:
                if self.expect_rate_limit: # if rate limit is expected, then return False
                    return False
                raise RuntimeError("failed to parse json from " + self.task_address + "/agent-scheduler/v1/queue" +
                                   f", {response.status_code}, {response.text}") from exc
            if 'current_task_id' not in req_json.keys() or 'pending_tasks' not in req_json.keys():
                raise RuntimeError(f"Parsed json from {self.task_address + '/agent-scheduler/v1/queue'} does not contain 'current_task_id' or 'pending_tasks', {response.status_code}, {response.text}")
            if self.task_id == req_json.get('current_task_id', None): # if current task is self.task_id, then return False
                return False
            elif any((self.task_id == task["api_task_id"] for task in req_json["pending_tasks"])):
                return False
            else:
                self._wait_between_calls(check_delay)
                result_response = self.session.get(self.task_address + "/agent-scheduler/v1/results/" + self.task_id)
                if result_response.status_code != 200:
                    raise RuntimeError(f"task id {self.task_id} is not found in queue or results, " +str(result_response.status_code), result_response.text)
                if not result_response.json().get('success', False):
                    # check 'Task is pending' or 'Task is running'
                    if result_response.json().get('message', '') == 'Task is pending' or result_response.json().get('message', '') == 'Task is running':
                        return False
                    elif result_response.json().get('message', '') == 'Task not found' or result_response.json().get('message', '') == 'Task result is not available':
                        raise ValueError(f"task id {self.task_id} has failed, " +str(result_response.status_code), result_response.text)
                    else:
                        raise RuntimeError(f"task id {self.task_id} has failed for unknown result, " +str(result_response.status_code), result_response.text)
                self.image = result_response.json()['data'][0]['image']
                self.images = [img['image'] for img in result_response.json()['data']]
                self.infotexts = [img.get('infotext') for img in result_response.json()['data']]
                self.terminated = True
                self.task_address = ""
                return True
        else:
            return True
        
    def _wait_between_calls(self, seconds=1):
        import time
        time.sleep(seconds)

class ControlNetUnit:
    def __init__(
        self,
        input_image: Image = None,
        mask: Image = None,
        module: str = "none",
        model: str = "None",
        weight: float = 1.0,
        resize_mode: str = "Resize and Fill",
        lowvram: bool = False,
        processor_res: int = 512,
        threshold_a: float = 64,
        threshold_b: float = 64,
        guidance: float = 1.0,
        guidance_start: float = 0.0,
        guidance_end: float = 1.0,
        control_mode: int = 0,
        pixel_perfect: bool = False,
        guessmode: int = None,  # deprecated: use control_mode
    ):
        self.input_image = input_image
        self.mask = mask
        self.module = module
        self.model = model
        self.weight = weight
        self.resize_mode = resize_mode
        self.lowvram = lowvram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance = guidance
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        if guessmode:
            print(
                "ControlNetUnit guessmode is deprecated. Please use control_mode instead."
            )
            control_mode = guessmode
        self.control_mode = control_mode
        self.pixel_perfect = pixel_perfect

    def to_dict(self):
        return {
            "input_image": raw_b64_img(self.input_image) if self.input_image else "",
            "mask": raw_b64_img(self.mask) if self.mask is not None else None,
            "module": self.module,
            "model": self.model,
            "weight": self.weight,
            "resize_mode": self.resize_mode,
            "lowvram": self.lowvram,
            "processor_res": self.processor_res,
            "threshold_a": self.threshold_a,
            "threshold_b": self.threshold_b,
            "guidance": self.guidance,
            "guidance_start": self.guidance_start,
            "guidance_end": self.guidance_end,
            "control_mode": self.control_mode,
            "pixel_perfect": self.pixel_perfect,
        }


def b64_img(image: Image) -> str:
    return  "data:image/png;base64," + raw_b64_img(image)

def raw_b64_img(image: Image) -> str:
    # XXX controlnet only accepts RAW base64 without headers
    with io.BytesIO() as output_bytes:
        metadata = None
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                if metadata is None:
                    metadata = PngImagePlugin.PngInfo()
                metadata.add_text(key, value)
        image.save(output_bytes, format="PNG", pnginfo=metadata)

        bytes_data = output_bytes.getvalue()

    return str(base64.b64encode(bytes_data), "utf-8")


class WebUIApi:
    has_controlnet = False

    def __init__(
        self,
        host="127.0.0.1",
        port=7860,
        baseurl=None,
        sampler="Euler a",
        steps=20,
        use_https=False,
        username=None,
        password=None,
    ):
        if baseurl is None:
            if use_https:
                baseurl = f"https://{host}:{port}/sdapi/v1"
            else:
                baseurl = f"http://{host}:{port}/sdapi/v1"
        self.real_url = baseurl.split('/sdapi/v1')[0]
        self.baseurl = baseurl
        self.default_sampler = sampler
        self.default_steps = steps

        self.session = requests.Session()

        if username and password:
            self.set_auth(username, password)
        else:
            self.check_controlnet()
            
        # test
        

    def check_controlnet(self):
        try:
            scripts = self.get_scripts()
            self.has_controlnet = "controlnet m2m" in scripts["txt2img"]
        except:
            pass

    def set_auth(self, username, password):
        self.session.auth = (username, password)
        self.check_controlnet()

    def _to_api_result(self, response):
        if response.status_code != 200:
            raise RuntimeError(response.status_code, response.text)

        r = response.json()
        # if {"task_id" : "string"} format, wrap 
        if "task_id" in r.keys():
            # remove '/sdapi/v1' from baseurl
            task_address = self.baseurl.split('/sdapi/v1')[0]
            return QueuedTaskResult(r["task_id"], task_address=task_address, session=self.session)
        images = []
        if "images" in r.keys():
            images = [Image.open(io.BytesIO(base64.b64decode(i))) for i in r["images"]]
        elif "image" in r.keys():
            images = [Image.open(io.BytesIO(base64.b64decode(r["image"])))]

        info = ""
        if "info" in r.keys():
            try:
                info = json.loads(r["info"])
            except:
                info = r["info"]
        elif "html_info" in r.keys():
            info = r["html_info"]
        elif "caption" in r.keys():
            info = r["caption"]

        parameters = ""
        if "parameters" in r.keys():
            parameters = r["parameters"]

        return WebUIApiResult(images, parameters, info)

    async def _to_api_result_async(self, response):
        if response.status != 200:
            raise RuntimeError(response.status, await response.text)

        r = await response.json()
        if "task_id" in r.keys():
            return QueuedTaskResult(r["task_id"], self.baseurl.split('/sdapi/v1')[0], session=self.session)
        images = []
        if "images" in r.keys():
            images = [Image.open(io.BytesIO(base64.b64decode(i))) for i in r["images"]]
        elif "image" in r.keys():
            images = [Image.open(io.BytesIO(base64.b64decode(r["image"])))]

        info = ""
        if "info" in r.keys():
            try:
                info = json.loads(r["info"])
            except:
                info = r["info"]
        elif "html_info" in r.keys():
            info = r["html_info"]
        elif "caption" in r.keys():
            info = r["caption"]

        parameters = ""
        if "parameters" in r.keys():
            parameters = r["parameters"]

        return WebUIApiResult(images, parameters, info)

    def txt2img(
        self,
        enable_hr=False,
        denoising_strength=0.7,
        firstphase_width=0,
        firstphase_height=0,
        hr_scale=2,
        hr_upscaler=HiResUpscaler.Latent,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0.0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name=None,  # use this instead of sampler_index
        batch_size=1,
        n_iter=1,
        steps=None,
        cfg_scale=7.0,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings={},
        override_settings_restore_afterwards=True,
        script_args=None,  # List of arguments for the script "script_name"
        script_name=None,
        send_images=True,
        save_images=False,
        alwayson_scripts={},
        controlnet_units: List[ControlNetUnit] = [],
        sampler_index=None,  # deprecated: use sampler_name
        use_deprecated_controlnet=False,
        use_async=False,
    ):
        if sampler_index is None:
            sampler_index = self.default_sampler
        if sampler_name is None:
            sampler_name = self.default_sampler
        if steps is None:
            steps = self.default_steps
        if script_args is None:
            script_args = []
        payload = {
            "enable_hr": enable_hr,
            "hr_scale": hr_scale,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_second_pass_steps,
            "hr_resize_x": hr_resize_x,
            "hr_resize_y": hr_resize_y,
            "denoising_strength": denoising_strength,
            "firstphase_width": firstphase_width,
            "firstphase_height": firstphase_height,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "seed_resize_from_h": seed_resize_from_h,
            "seed_resize_from_w": seed_resize_from_w,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "do_not_save_samples": do_not_save_samples,
            "do_not_save_grid": do_not_save_grid,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "s_churn": s_churn,
            "s_tmax": s_tmax,
            "s_tmin": s_tmin,
            "s_noise": s_noise,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
            "sampler_name": sampler_name,
            "sampler_index": sampler_index,
            "script_name": script_name,
            "script_args": script_args,
            "send_images": send_images,
            "save_images": save_images,
            "alwayson_scripts": alwayson_scripts,
        }

        if use_deprecated_controlnet and controlnet_units and len(controlnet_units) > 0:
            payload["controlnet_units"] = [x.to_dict() for x in controlnet_units]
            return self.custom_post(
                "controlnet/txt2img", payload=payload, use_async=use_async
            )

        if controlnet_units and len(controlnet_units) > 0:
            payload["alwayson_scripts"]["ControlNet"] = {
                "args": [x.to_dict() for x in controlnet_units]
            }
        elif self.has_controlnet:
            # workaround : if not passed, webui will use previous args!
            payload["alwayson_scripts"]["ControlNet"] = {"args": []}

        return self.post_and_get_api_result(
            f"{self.baseurl}/txt2img", payload, use_async
        )

    def txt2img_task(
        self,
        enable_hr=False,
        denoising_strength=0.7,
        firstphase_width=0,
        firstphase_height=0,
        hr_scale=2,
        hr_upscaler=HiResUpscaler.Latent,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        prompt="",
        styles=list(),
        seed=-1,
        subseed=-1,
        subseed_strength=0.0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name=None,  # use this instead of sampler_index
        batch_size=1,
        n_iter=1,
        steps=None,
        cfg_scale=7.0,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings=dict(),
        override_settings_restore_afterwards=True,
        script_args=None,  # List of arguments for the script "script_name"
        script_name=None,
        send_images=True,
        save_images=False,
        alwayson_scripts=dict(),
        controlnet_units: List[ControlNetUnit] = list(),
        sampler_index=None,  # deprecated: use sampler_name
        use_deprecated_controlnet=False,
        use_async=False,
        checkpoint_name=None,
    ):
        """
        This function is similar to txt2img, but it returns a QueuedTaskResult instead of WebUIApiResult.
        You can also pass checkpoint_name to use a specific checkpoint.
        """
        if sampler_index is None:
            sampler_index = self.default_sampler
        if sampler_name is None:
            sampler_name = self.default_sampler
        if steps is None:
            steps = self.default_steps
        if script_args is None:
            script_args = []
        payload = {
            "enable_hr": enable_hr,
            "hr_scale": hr_scale,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_second_pass_steps,
            "hr_resize_x": hr_resize_x,
            "hr_resize_y": hr_resize_y,
            "denoising_strength": denoising_strength,
            "firstphase_width": firstphase_width,
            "firstphase_height": firstphase_height,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "seed_resize_from_h": seed_resize_from_h,
            "seed_resize_from_w": seed_resize_from_w,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "do_not_save_samples": do_not_save_samples,
            "do_not_save_grid": do_not_save_grid,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "s_churn": s_churn,
            "s_tmax": s_tmax,
            "s_tmin": s_tmin,
            "s_noise": s_noise,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
            "sampler_name": sampler_name,
            "sampler_index": sampler_index,
            "script_name": script_name,
            "script_args": script_args,
            "send_images": send_images,
            "save_images": save_images,
            "alwayson_scripts": alwayson_scripts,
        }
        if checkpoint_name is not None:
            payload["checkpoint_name"] = checkpoint_name

        if use_deprecated_controlnet:
            raise RuntimeError("use_deprecated_controlnet is not supported for txt2img_task")
        if controlnet_units and len(controlnet_units) > 0:
            payload["alwayson_scripts"]["ControlNet"] = {
                "args": [x.to_dict() for x in controlnet_units]
            }
        elif self.has_controlnet:
            # workaround : if not passed, webui will use previous args!
            payload["alwayson_scripts"]["ControlNet"] = {"args": []}

        return self.post_and_get_api_result(
            f"{self.baseurl.split('/sdapi/v1')[0]}" + "/agent-scheduler/v1/queue/txt2img", payload, use_async
        )
        
    def post_and_get_api_result(self, url, json, use_async):
        if use_async:
            import asyncio

            return asyncio.ensure_future(self.async_post(url=url, json=json))
        else:
            response = self.session.post(url=url, json=json)
            return self._to_api_result(response)
        
    def resume_agent_scheduler(self):
        # post agent-scheduler/v1/queue/resume
        target_url = self.real_url + "/agent-scheduler/v1/queue/resume"
        return self.session.post(target_url)
    
    def pause_agent_scheduler(self):
        # post agent-scheduler/v1/queue/pause
        target_url = self.real_url + "/agent-scheduler/v1/queue/pause"
        return self.session.post(target_url)

    async def async_post(self, url, json):
        import aiohttp

        async with aiohttp.ClientSession() as session:
            auth = aiohttp.BasicAuth(self.session.auth[0], self.session.auth[1]) if self.session.auth else None
            async with session.post(url, json=json, auth=auth) as response:
                return await self._to_api_result_async(response)
            
    def set_overwrite(self, overwrite:bool):
        # /set_overwrite
        target_url = self.real_url + "/upload_options/overwrite"
        return self.session.post(target_url, data={"overwrite": overwrite})

    def upload_lora(self, lora_file_path, lora_target_path:str="", custom_name:str=""):
        """
        Upload lora model to server.
        example:
        lora_file_path = "lora.pt"
        lora_target_path = 'folder_name"
        """
        target_url = self.real_url + "/upload_lora_model"
        import os
        assert os.path.exists(lora_file_path), "lora file not found at " + lora_file_path
        files = {'file': open(lora_file_path, 'rb')}
        data = {}
        if lora_target_path != "":
            data["lora_path"] = lora_target_path
        if custom_name != "":
            data["custom_name"] = custom_name
        return self.session.post(target_url, files=files, data=data)
        
    def upload_sd_model(self, sd_file_path, sd_target_path:str="", custom_name:str=""):
        # /upload_sd_model
        
        target_url = self.real_url + "/upload_sd_model"
        import os
        assert os.path.exists(sd_file_path), "sd file not found at " + sd_file_path
        files = {'file': open(sd_file_path, 'rb')}
        data = {}
        if sd_target_path != "":
            data["sd_path"] = sd_target_path
        if custom_name != "":
            data["custom_name"] = custom_name
        return self.session.post(target_url, files=files, data=data)
        
    def upload_vae_model(self, vae_file_path, vae_target_path:str="", custom_name:str=""):
        # /upload_vae_model
        
        target_url = self.real_url + "/upload_vae_model"
        import os
        assert os.path.exists(vae_file_path), "vae file not found at " + vae_file_path
        files = {'file': open(vae_file_path, 'rb')}
        data = {}
        if vae_target_path != "":
            data["vae_path"] = vae_target_path
        if custom_name != "":
            data["custom_name"] = custom_name
        return self.session.post(target_url, files=files, data=data)
        

    def upload_textual_inversion(self, textual_inversion_file_path, textual_inversion_target_path:str="", custom_name:str=""):
        # /upload_textual_inversion
        
        target_url = self.real_url + "/upload_embedding"
        import os
        assert os.path.exists(textual_inversion_file_path), "textual_inversion file not found at " + textual_inversion_file_path
        files = {'file': open(textual_inversion_file_path, 'rb')}
        data = {}
        if textual_inversion_target_path != "":
            data["textual_inversion_path"] = textual_inversion_target_path
        if custom_name != "":
            data["custom_name"] = custom_name
        return self.session.post(target_url, files=files, data=data)
        
    def upload_dynamic_prompts(self, dynamic_prompts_file_path, dynamic_prompts_target_path:str=""):
        """
        Upload dynamic prompts to server.
        """
        # /upload/dynamic_prompts
        
        target_url = self.real_url + "/upload_dynamic_prompts"
        import os
        assert os.path.exists(dynamic_prompts_file_path), "dynamic_prompts file not found at " + dynamic_prompts_file_path
        #curl -X POST -F "text=@C:\\Users\\UserName\\Downloads\\test.txt" -F "path=dynamic_prompts/test.txt" http://localhost:7860/upload_dynamic_prompts
        files = {'text': open(dynamic_prompts_file_path, 'rb')}
        data = {}
        if dynamic_prompts_target_path != "":
            data["path"] = dynamic_prompts_target_path
        return self.session.post(target_url, files=files, data=data)
    
    def check_uploader_ping(self, target_api_address:str = ""):
        """
        Checks if target_api_address is running uploader
        
        @param target_api_address: address of target api or empty string to check self
        
        @throws RuntimeError if target_api_address is not running uploader
        """
        # ping /uploader/ping to check if uploader is running
        target_api_ping_url = target_api_address + "/uploader/ping"
        if target_api_address == "":
            # check self
            target_api_ping_url = self.real_url + "/uploader/ping"
        result = self.session.get(target_api_ping_url)
        # check response
        if result.status_code != 200:
            result_error = result.text
            raise RuntimeError("target_api_address is not running uploader, " + str(result.status_code) + " " + result_error)
        
    def request_download_controlnets(self, model_type:str = 'v11'):
        # model type can be 'v11' or 'xl'
        # /download_controlnet_models/v11
        # curl -X POST http://127.0.0.1:7860/download_controlnet_models/v11
        target_url = self.real_url + "/download_controlnet_models/" + model_type
        return self.session.post(target_url) # no data
        
        
    def send_single_sync_request(self, target_api_address:str, target_auth:str, model_type:str, model_path:str=""):
        """
        Sends single model sync request to target_api_address
        """
        if model_type not in ["sd_model", "lora_model", "vae_model", "embedding"]:
            raise RuntimeError("model_type must be one of sd_model, lora_model, vae_model, embedding, but given " + model_type)
        if target_api_address == "":
            raise RuntimeError("target_api_address must be specified")
        target_url = self.real_url + "/sync/" + model_type
        data = {}
        if target_api_address != "":
            data["target_api_address"] = target_api_address
        if target_auth != "":
            data["auth"] = target_auth
        if model_path != "":
            data["model_path"] = model_path
        else:
            raise RuntimeError("model_path must be specified")
        return self.session.post(target_url, data=data)
    
    def send_model_type_sync_request(self, target_api_address:str, target_auth:str, model_type:str):
        """
        Sends model type sync request to target_api_address
        """
        # curl -X POST -F "target_api_address=http://test.api.address/" http://127.0.0.1:7860/sync/all_sd_models
        if model_type not in ["all_sd_models", "all_lora_models", "all_vae_models", "all_models"]:
            raise RuntimeError("model_type must be one of all_sd_models, all_lora_models, all_vae_models, all_models, but given " + model_type)
        target_url = self.real_url + "/sync/" + model_type
        data = {}
        if target_api_address != "":
            data["target_api_address"] = target_api_address
        if target_auth != "":
            data["auth"] = target_auth
        return self.session.post(target_url, data=data)
    
    def send_lora_sync_request(self, target_api_address:str, target_auth:str, model_path:str=""):
        """
        Sends lora sync request to target_api_address
        """
        # curl -X POST -F "target_api_address=http://test.api.address/" -F "model_path=test/test.safetensors" -F"auth=username:password" http://
        return self.send_single_sync_request(target_api_address, target_auth, "lora_model", model_path)
    
    def send_sd_sync_request(self, target_api_address:str, target_auth:str, model_path:str=""):
        """
        Sends sd sync request to target_api_address
        """
        return self.send_single_sync_request(target_api_address, target_auth, "sd_model", model_path)
    
    def send_vae_sync_request(self, target_api_address:str, target_auth:str, model_path:str=""):
        """
        Sends vae sync request to target_api_address
        """
        return self.send_single_sync_request(target_api_address, target_auth, "vae_model", model_path)
    def send_embedding_sync_request(self, target_api_address:str, target_auth:str, model_path:str=""):
        """
        Sends embedding sync request to target_api_address
        """
        return self.send_single_sync_request(target_api_address, target_auth, "embedding", model_path)
    def send_all_sd_models_sync_request(self, target_api_address:str, target_auth:str):
        """
        Sends all_sd_models sync request to target_api_address
        """
        return self.send_model_type_sync_request(target_api_address, target_auth, "all_sd_models")
    def send_all_lora_models_sync_request(self, target_api_address:str, target_auth:str):
        """
        Sends all_lora_models sync request to target_api_address
        """
        return self.send_model_type_sync_request(target_api_address, target_auth, "all_lora_models")
    def send_all_vae_models_sync_request(self, target_api_address:str, target_auth:str):
        """
        Sends all_vae_models sync request to target_api_address
        """
        return self.send_model_type_sync_request(target_api_address, target_auth, "all_vae_models")
    def send_all_models_sync_request(self, target_api_address:str, target_auth:str):
        """
        Sends all_models sync request to target_api_address
        """
        return self.send_model_type_sync_request(target_api_address, target_auth, "all_models")
        
    
    
    def remove_sd_model(self, sd_target_path:str=""):
        # /remove_sd_model
        
        target_url = self.real_url + "/delete/sd_model"
        data = {}
        if sd_target_path != "":
            data["model_path"] = sd_target_path
        return self.session.post(target_url, data=data)
    
    def remove_lora_model(self, lora_target_path:str=""):
        # /remove_lora_model
        
        target_url = self.real_url + "/delete/lora_model"
        data = {}
        if lora_target_path != "":
            data["model_path"] = lora_target_path
        return self.session.post(target_url, data=data)
    
    def remove_vae_model(self, vae_target_path:str=""):
        # /remove_vae_model
        
        target_url = self.real_url + "/delete/vae_model"
        data = {}
        if vae_target_path != "":
            data["model_path"] = vae_target_path
        return self.session.post(target_url, data=data)
    
    def remove_textual_inversion(self, textual_inversion_target_path:str=""):
        # /remove_textual_inversion
        
        target_url = self.real_url + "/delete/embedding"
        data = {}
        if textual_inversion_target_path != "":
            data["model_path"] = textual_inversion_target_path
        return self.session.post(target_url, data=data)
        
    def query_hash_all(self, subpath:str=""):
        # /models/query_hash_sd_all
        target_url = self.real_url + "/models/query_hash_all"
        # post path=subpath
        payload = {"path": subpath}
        result = self.session.post(target_url, json=payload)
        # curl -X POST -H "Content-Type: application/json" -d "{\"path\":\""}" http://127.0.0.1:7860/models/query_hash_sd_all
        result_json = result.json()
        if result.status_code != 200:
            raise RuntimeError(result.status_code, result.text)
        return result_json
    
    def query_hash_loras(self, subpath:str=""):
        # /models/query_hash_sd_all
        target_url = self.real_url + "/models/query_hash_lora_all"
        # post path=subpath
        payload = {"path": subpath}
        result = self.session.post(target_url, json=payload)
        # curl -X POST -H "Content-Type: application/json" -d "{\"path\":\""}" http://127.0.0.1:7860/models/query_hash_sd_all
        result_json = result.json()
        if result.status_code != 200:
            raise RuntimeError(result.status_code, result.text)
        return result_json 
    
    def img2img(
        self,
        images=[],  # list of PIL Image
        resize_mode=0,
        denoising_strength=0.75,
        image_cfg_scale=1.5,
        mask_image=None,  # PIL Image mask
        mask_blur=4,
        inpainting_fill=0,
        inpaint_full_res=True,
        inpaint_full_res_padding=0,
        inpainting_mask_invert=0,
        initial_noise_multiplier=1,
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name=None,  # use this instead of sampler_index
        batch_size=1,
        n_iter=1,
        steps=None,
        cfg_scale=7.0,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings={},
        override_settings_restore_afterwards=True,
        script_args=None,  # List of arguments for the script "script_name"
        sampler_index=None,  # deprecated: use sampler_name
        include_init_images=False,
        script_name=None,
        send_images=True,
        save_images=False,
        alwayson_scripts={},
        controlnet_units: List[ControlNetUnit] = [],
        use_deprecated_controlnet=False,
        use_async=False,
    ):
        if sampler_name is None:
            sampler_name = self.default_sampler
        if sampler_index is None:
            sampler_index = self.default_sampler
        if steps is None:
            steps = self.default_steps
        if script_args is None:
            script_args = []

        payload = {
            "init_images": [b64_img(x) for x in images],
            "resize_mode": resize_mode,
            "denoising_strength": denoising_strength,
            "mask_blur": mask_blur,
            "inpainting_fill": inpainting_fill,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpainting_mask_invert": inpainting_mask_invert,
            "initial_noise_multiplier": initial_noise_multiplier,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "seed_resize_from_h": seed_resize_from_h,
            "seed_resize_from_w": seed_resize_from_w,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "do_not_save_samples": do_not_save_samples,
            "do_not_save_grid": do_not_save_grid,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "s_churn": s_churn,
            "s_tmax": s_tmax,
            "s_tmin": s_tmin,
            "s_noise": s_noise,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
            "sampler_name": sampler_name,
            "sampler_index": sampler_index,
            "include_init_images": include_init_images,
            "script_name": script_name,
            "script_args": script_args,
            "send_images": send_images,
            "save_images": save_images,
            "alwayson_scripts": alwayson_scripts,
        }
        if mask_image is not None:
            payload["mask"] = b64_img(mask_image)

        if use_deprecated_controlnet and controlnet_units and len(controlnet_units) > 0:
            payload["controlnet_units"] = [x.to_dict() for x in controlnet_units]
            return self.custom_post(
                "controlnet/img2img", payload=payload, use_async=use_async
            )

        if controlnet_units and len(controlnet_units) > 0:
            payload["alwayson_scripts"]["ControlNet"] = {
                "args": [x.to_dict() for x in controlnet_units]
            }
        elif self.has_controlnet:
            payload["alwayson_scripts"]["ControlNet"] = {"args": []}

        return self.post_and_get_api_result(
            f"{self.baseurl}/img2img", payload, use_async
        )

    def img2img_task(
        self,
        images=[],  # list of PIL Image
        resize_mode=0,
        denoising_strength=0.75,
        image_cfg_scale=1.5,
        mask_image=None,  # PIL Image mask
        mask_blur=4,
        inpainting_fill=0,
        inpaint_full_res=True,
        inpaint_full_res_padding=0,
        inpainting_mask_invert=0,
        initial_noise_multiplier=1,
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name=None,  # use this instead of sampler_index
        batch_size=1,
        n_iter=1,
        steps=None,
        cfg_scale=7.0,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings={},
        override_settings_restore_afterwards=True,
        script_args=None,  # List of arguments for the script "script_name"
        sampler_index=None,  # deprecated: use sampler_name
        include_init_images=False,
        script_name=None,
        send_images=True,
        save_images=False,
        alwayson_scripts={},
        controlnet_units: List[ControlNetUnit] = [],
        use_deprecated_controlnet=False,
        use_async=False,
        checkpoint_name=None,
    ):
        if sampler_name is None:
            sampler_name = self.default_sampler
        if sampler_index is None:
            sampler_index = self.default_sampler
        if steps is None:
            steps = self.default_steps
        if script_args is None:
            script_args = []

        payload = {
            "init_images": [b64_img(x) for x in images],
            "resize_mode": resize_mode,
            "denoising_strength": denoising_strength,
            "mask_blur": mask_blur,
            "inpainting_fill": inpainting_fill,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpainting_mask_invert": inpainting_mask_invert,
            "initial_noise_multiplier": initial_noise_multiplier,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "seed_resize_from_h": seed_resize_from_h,
            "seed_resize_from_w": seed_resize_from_w,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "do_not_save_samples": do_not_save_samples,
            "do_not_save_grid": do_not_save_grid,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "s_churn": s_churn,
            "s_tmax": s_tmax,
            "s_tmin": s_tmin,
            "s_noise": s_noise,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
            "sampler_name": sampler_name,
            "sampler_index": sampler_index,
            "include_init_images": include_init_images,
            "script_name": script_name,
            "script_args": script_args,
            "send_images": send_images,
            "save_images": save_images,
            "alwayson_scripts": alwayson_scripts,
        }
        if checkpoint_name is not None:
            payload["checkpoint_name"] = checkpoint_name
        if mask_image is not None:
            payload["mask"] = b64_img(mask_image)

        if use_deprecated_controlnet and controlnet_units and len(controlnet_units) > 0:
            payload["controlnet_units"] = [x.to_dict() for x in controlnet_units]
            return self.custom_post(
                "controlnet/img2img", payload=payload, use_async=use_async
            )

        if controlnet_units and len(controlnet_units) > 0:
            payload["alwayson_scripts"]["ControlNet"] = {
                "args": [x.to_dict() for x in controlnet_units]
            }
        elif self.has_controlnet:
            payload["alwayson_scripts"]["ControlNet"] = {"args": []}

        return self.post_and_get_api_result(
            f"{self.baseurl.split('/sdapi/v1')[0]}" + "/agent-scheduler/v1/queue/img2img", payload, use_async
        )
    def extra_single_image(
        self,
        image,  # PIL Image
        resize_mode=0,
        show_extras_results=True,
        gfpgan_visibility=0,
        codeformer_visibility=0,
        codeformer_weight=0,
        upscaling_resize=2,
        upscaling_resize_w=512,
        upscaling_resize_h=512,
        upscaling_crop=True,
        upscaler_1="None",
        upscaler_2="None",
        extras_upscaler_2_visibility=0,
        upscale_first=False,
        use_async=False,
    ):
        payload = {
            "resize_mode": resize_mode,
            "show_extras_results": show_extras_results,
            "gfpgan_visibility": gfpgan_visibility,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "upscaling_resize": upscaling_resize,
            "upscaling_resize_w": upscaling_resize_w,
            "upscaling_resize_h": upscaling_resize_h,
            "upscaling_crop": upscaling_crop,
            "upscaler_1": upscaler_1,
            "upscaler_2": upscaler_2,
            "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
            "upscale_first": upscale_first,
            "image": b64_img(image),
        }

        return self.post_and_get_api_result(
            f"{self.baseurl}/extra-single-image", payload, use_async
        )

    def extra_batch_images(
        self,
        images,  # list of PIL images
        name_list=None,  # list of image names
        resize_mode=0,
        show_extras_results=True,
        gfpgan_visibility=0,
        codeformer_visibility=0,
        codeformer_weight=0,
        upscaling_resize=2,
        upscaling_resize_w=512,
        upscaling_resize_h=512,
        upscaling_crop=True,
        upscaler_1="None",
        upscaler_2="None",
        extras_upscaler_2_visibility=0,
        upscale_first=False,
        use_async=False,
    ):
        if name_list is not None:
            if len(name_list) != len(images):
                raise RuntimeError("len(images) != len(name_list)")
        else:
            name_list = [f"image{i + 1:05}" for i in range(len(images))]
        images = [b64_img(x) for x in images]

        image_list = []
        for name, image in zip(name_list, images):
            image_list.append({"data": image, "name": name})

        payload = {
            "resize_mode": resize_mode,
            "show_extras_results": show_extras_results,
            "gfpgan_visibility": gfpgan_visibility,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "upscaling_resize": upscaling_resize,
            "upscaling_resize_w": upscaling_resize_w,
            "upscaling_resize_h": upscaling_resize_h,
            "upscaling_crop": upscaling_crop,
            "upscaler_1": upscaler_1,
            "upscaler_2": upscaler_2,
            "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
            "upscale_first": upscale_first,
            "imageList": image_list,
        }

        return self.post_and_get_api_result(
            f"{self.baseurl}/extra-batch-images", payload, use_async
        )

    # XXX 500 error (2022/12/26)
    def png_info(self, image):
        payload = {
            "image": b64_img(image),
        }

        response = self.session.post(url=f"{self.baseurl}/png-info", json=payload)
        return self._to_api_result(response)

    """
    :param image pass base64 encoded image or PIL Image
    :param model "clip" or "deepdanbooru"
    """
    def interrogate(self, image, model="clip"):
        payload = {
            "image": b64_img(image) if isinstance(image, Image.Image) else image,
            "model": model,
        }

        response = self.session.post(url=f"{self.baseurl}/interrogate", json=payload)
        return self._to_api_result(response)

    def interrupt(self):
        response = self.session.post(url=f"{self.baseurl}/interrupt")
        return response.json()

    def skip(self):
        response = self.session.post(url=f"{self.baseurl}/skip")
        return response.json()

    def get_options(self):
        response = self.session.get(url=f"{self.baseurl}/options")
        return response.json()

    def set_options(self, options):
        response = self.session.post(url=f"{self.baseurl}/options", json=options)
        return response.json()

    def get_progress(self):
        response = self.session.get(url=f"{self.baseurl}/progress")
        return response.json()

    def get_cmd_flags(self):
        response = self.session.get(url=f"{self.baseurl}/cmd-flags")
        return response.json()

    def get_samplers(self):
        response = self.session.get(url=f"{self.baseurl}/samplers")
        return response.json()

    def get_sd_vae(self):
        response = self.session.get(url=f"{self.baseurl}/sd-vae")
        return response.json()

    def get_upscalers(self):
        response = self.session.get(url=f"{self.baseurl}/upscalers")
        return response.json()

    def get_latent_upscale_modes(self):
        response = self.session.get(url=f"{self.baseurl}/latent-upscale-modes")
        return response.json()

    def get_loras(self):
        response = self.session.get(url=f"{self.baseurl}/loras")
        return response.json()

    def get_sd_models(self):
        response = self.session.get(url=f"{self.baseurl}/sd-models")
        return response.json()

    def get_hypernetworks(self):
        response = self.session.get(url=f"{self.baseurl}/hypernetworks")
        return response.json()

    def get_face_restorers(self):
        response = self.session.get(url=f"{self.baseurl}/face-restorers")
        return response.json()

    def get_realesrgan_models(self):
        response = self.session.get(url=f"{self.baseurl}/realesrgan-models")
        return response.json()

    def get_prompt_styles(self):
        response = self.session.get(url=f"{self.baseurl}/prompt-styles")
        return response.json()

    def get_artist_categories(self):  # deprecated ?
        response = self.session.get(url=f"{self.baseurl}/artist-categories")
        return response.json()

    def get_artists(self):  # deprecated ?
        response = self.session.get(url=f"{self.baseurl}/artists")
        return response.json()

    def refresh_checkpoints(self):
        response = self.session.post(url=f"{self.baseurl}/refresh-checkpoints")
        return response.json()
    
    def refresh_loras(self) -> None: # returns null, be careful
        response = self.session.post(url=f"{self.baseurl}/refresh-loras")
        return response.json()

    def get_scripts(self):
        response = self.session.get(url=f"{self.baseurl}/scripts")
        return response.json()

    def get_scripts_info(self):
        response = self.session.get(url=f"{self.baseurl}/script-info")
        return response.json()

    def get_embeddings(self):
        response = self.session.get(url=f"{self.baseurl}/embeddings")
        return response.json()

    def get_memory(self):
        response = self.session.get(url=f"{self.baseurl}/memory")
        return response.json()

    def get_endpoint(self, endpoint, baseurl):
        if baseurl:
            return f"{self.baseurl}/{endpoint}"
        else:
            from urllib.parse import urlparse, urlunparse

            parsed_url = urlparse(self.baseurl)
            basehost = parsed_url.netloc
            parsed_url2 = (parsed_url[0], basehost, endpoint, "", "", "")
            return urlunparse(parsed_url2)

    def custom_get(self, endpoint, baseurl=False):
        url = self.get_endpoint(endpoint, baseurl)
        response = self.session.get(url=url)
        return response.json()

    def custom_post(self, endpoint, payload={}, baseurl=False, use_async=False):
        url = self.get_endpoint(endpoint, baseurl)
        if use_async:
            import asyncio

            return asyncio.ensure_future(self.async_post(url=url, json=payload))
        else:
            response = self.session.post(url=url, json=payload)
            return self._to_api_result(response)

    def controlnet_version(self):
        r = self.custom_get("controlnet/version")
        return r["version"]

    def controlnet_model_list(self):
        r = self.custom_get("controlnet/model_list")
        return r["model_list"]

    def controlnet_module_list(self):
        r = self.custom_get("controlnet/module_list")
        return r["module_list"]

    def controlnet_detect(
        self, images, module="none", processor_res=512, threshold_a=64, threshold_b=64
    ):
        input_images = [b64_img(x) for x in images]
        payload = {
            "controlnet_module": module,
            "controlnet_input_images": input_images,
            "controlnet_processor_res": processor_res,
            "controlnet_threshold_a": threshold_a,
            "controlnet_threshold_b": threshold_b,
        }
        r = self.custom_post("controlnet/detect", payload=payload)
        return r

    def util_get_model_names(self):
        return sorted([x["title"] for x in self.get_sd_models()])

    def util_set_model(self, name, find_closest=True):
        if find_closest:
            name = name.lower()
        models = self.util_get_model_names()
        found_model = None
        if name in models:
            found_model = name
        elif find_closest:
            import difflib

            def str_simularity(a, b):
                return difflib.SequenceMatcher(None, a, b).ratio()

            max_sim = 0.0
            max_model = models[0]
            for model in models:
                sim = str_simularity(name, model)
                if sim >= max_sim:
                    max_sim = sim
                    max_model = model
            found_model = max_model
        if found_model:
            print(f"loading {found_model}")
            options = {}
            options["sd_model_checkpoint"] = found_model
            self.set_options(options)
            print(f"model changed to {found_model}")
        else:
            print("model not found")

    def util_get_current_model(self):
        options = self.get_options()
        if ("sd_model_checkpoint" in options):
            return options["sd_model_checkpoint"]
        else:
            sd_models = self.get_sd_models()
            sd_model = [model for model in sd_models if model["sha256"] == options["sd_checkpoint_hash"]]
            return sd_model[0]["title"]

    def util_wait_for_ready(self, check_interval=5.0):
        import time

        while True:
            result = self.get_progress()
            progress = result["progress"]
            job_count = result["state"]["job_count"]
            if progress == 0.0 and job_count == 0:
                break
            else:
                print(f"[WAIT]: progress = {progress:.4f}, job_count = {job_count}")
                time.sleep(check_interval)


## Interface for extensions


# https://github.com/mix1009/model-keyword
@dataclass
class ModelKeywordResult:
    keywords: list
    model: str
    oldhash: str
    match_source: str


class ModelKeywordInterface:
    def __init__(self, webuiapi):
        self.api = webuiapi

    def get_keywords(self):
        result = self.api.custom_get("model_keyword/get_keywords")
        keywords = result["keywords"]
        model = result["model"]
        oldhash = result["hash"]
        match_source = result["match_source"]
        return ModelKeywordResult(keywords, model, oldhash, match_source)





# https://github.com/Klace/stable-diffusion-webui-instruct-pix2pix
class InstructPix2PixInterface:
    def __init__(self, webuiapi):
        self.api = webuiapi

    def img2img(
        self,
        images=[],
        prompt: str = "",
        negative_prompt: str = "",
        output_batches: int = 1,
        sampler: str = "Euler a",
        steps: int = 20,
        seed: int = 0,
        randomize_seed: bool = True,
        text_cfg: float = 7.5,
        image_cfg: float = 1.5,
        randomize_cfg: bool = False,
        output_image_width: int = 512,
    ):
        init_images = [b64_img(x) for x in images]
        payload = {
            "init_images": init_images,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_batches": output_batches,
            "sampler": sampler,
            "steps": steps,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "text_cfg": text_cfg,
            "image_cfg": image_cfg,
            "randomize_cfg": randomize_cfg,
            "output_image_width": output_image_width,
        }
        return self.api.custom_post("instruct-pix2pix/img2img", payload=payload)


#https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg
class RemBGInterface:
    def __init__(self, webuiapi):
        self.api = webuiapi

    def rembg(
        self,
        input_image: str = "", #image string (?)
        model: str = 'u2net',  #[None, 'u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg','silueta','isnet-general-use','isnet-anime']
        return_mask: bool = False,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10
    ):

        payload = {
            "input_image": b64_img(input_image),
            "model": model,
            "return_mask": return_mask,
            "alpha_matting":  alpha_matting,
            "alpha_matting_foreground_threshold": alpha_matting_foreground_threshold,
            "alpha_matting_background_threshold": alpha_matting_background_threshold,
            "alpha_matting_erode_size": alpha_matting_erode_size
        }
        return self.api.custom_post("rembg", payload=payload)


# https://github.com/Mikubill/sd-webui-controlnet
class ControlNetInterface:
    def __init__(self, webuiapi, show_deprecation_warning=True):
        self.api = webuiapi
        self.show_deprecation_warning = show_deprecation_warning

    def print_deprecation_warning(self):
        print(
            "ControlNetInterface txt2img/img2img is deprecated. Please use normal txt2img/img2img with controlnet_units param"
        )

    def txt2img(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        controlnet_input_image: [] = [],
        controlnet_mask: [] = [],
        controlnet_module: str = "",
        controlnet_model: str = "",
        controlnet_weight: float = 0.5,
        controlnet_resize_mode: str = "Scale to Fit (Inner Fit)",
        controlnet_lowvram: bool = False,
        controlnet_processor_res: int = 512,
        controlnet_threshold_a: int = 64,
        controlnet_threshold_b: int = 64,
        controlnet_guidance: float = 1.0,
        enable_hr: bool = False,  # hiresfix
        denoising_strength: float = 0.5,
        hr_scale: float = 1.5,
        hr_upscale: str = "Latent",
        guess_mode: bool = True,
        seed: int = -1,
        subseed: int = -1,
        subseed_strength: int = -1,
        sampler_index: str = "Euler a",
        batch_size: int = 1,
        n_iter: int = 1,  # Iteration
        steps: int = 20,
        cfg_scale: float = 7,
        width: int = 512,
        height: int = 512,
        restore_faces: bool = False,
        override_settings: Dict[str, Any] = None,
        override_settings_restore_afterwards: bool = True,
    ):
        if self.show_deprecation_warning:
            self.print_deprecation_warning()

        controlnet_input_image_b64 = [raw_b64_img(x) for x in controlnet_input_image]
        controlnet_mask_b64 = [raw_b64_img(x) for x in controlnet_mask]

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_input_image": controlnet_input_image_b64,
            "controlnet_mask": controlnet_mask_b64,
            "controlnet_module": controlnet_module,
            "controlnet_model": controlnet_model,
            "controlnet_weight": controlnet_weight,
            "controlnet_resize_mode": controlnet_resize_mode,
            "controlnet_lowvram": controlnet_lowvram,
            "controlnet_processor_res": controlnet_processor_res,
            "controlnet_threshold_a": controlnet_threshold_a,
            "controlnet_threshold_b": controlnet_threshold_b,
            "controlnet_guidance": controlnet_guidance,
            "enable_hr": enable_hr,
            "denoising_strength": denoising_strength,
            "hr_scale": hr_scale,
            "hr_upscale": hr_upscale,
            "guess_mode": guess_mode,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "sampler_index": sampler_index,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
        }
        return self.api.custom_post("controlnet/txt2img", payload=payload)

    def img2img(
        self,
        init_images: [] = [],
        mask: str = None,
        mask_blur: int = 30,
        inpainting_fill: int = 0,
        inpaint_full_res: bool = True,
        inpaint_full_res_padding: int = 1,
        inpainting_mask_invert: int = 1,
        resize_mode: int = 0,
        denoising_strength: float = 0.7,
        prompt: str = "",
        negative_prompt: str = "",
        controlnet_input_image: [] = [],
        controlnet_mask: [] = [],
        controlnet_module: str = "",
        controlnet_model: str = "",
        controlnet_weight: float = 1.0,
        controlnet_resize_mode: str = "Scale to Fit (Inner Fit)",
        controlnet_lowvram: bool = False,
        controlnet_processor_res: int = 512,
        controlnet_threshold_a: int = 64,
        controlnet_threshold_b: int = 64,
        controlnet_guidance: float = 1.0,
        guess_mode: bool = True,
        seed: int = -1,
        subseed: int = -1,
        subseed_strength: int = -1,
        sampler_index: str = "",
        batch_size: int = 1,
        n_iter: int = 1,  # Iteration
        steps: int = 20,
        cfg_scale: float = 7,
        width: int = 512,
        height: int = 512,
        restore_faces: bool = False,
        include_init_images: bool = True,
        override_settings: Dict[str, Any] = None,
        override_settings_restore_afterwards: bool = True,
    ):
        if self.show_deprecation_warning:
            self.print_deprecation_warning()

        init_images_b64 = [raw_b64_img(x) for x in init_images]
        controlnet_input_image_b64 = [raw_b64_img(x) for x in controlnet_input_image]
        controlnet_mask_b64 = [raw_b64_img(x) for x in controlnet_mask]

        payload = {
            "init_images": init_images_b64,
            "mask": raw_b64_img(mask) if mask else None,
            "mask_blur": mask_blur,
            "inpainting_fill": inpainting_fill,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpainting_mask_invert": inpainting_mask_invert,
            "resize_mode": resize_mode,
            "denoising_strength": denoising_strength,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_input_image": controlnet_input_image_b64,
            "controlnet_mask": controlnet_mask_b64,
            "controlnet_module": controlnet_module,
            "controlnet_model": controlnet_model,
            "controlnet_weight": controlnet_weight,
            "controlnet_resize_mode": controlnet_resize_mode,
            "controlnet_lowvram": controlnet_lowvram,
            "controlnet_processor_res": controlnet_processor_res,
            "controlnet_threshold_a": controlnet_threshold_a,
            "controlnet_threshold_b": controlnet_threshold_b,
            "controlnet_guidance": controlnet_guidance,
            "guess_mode": guess_mode,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "sampler_index": sampler_index,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "include_init_images": include_init_images,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
        }
        return self.api.custom_post("controlnet/img2img", payload=payload)

    def model_list(self):
        r = self.api.custom_get("controlnet/model_list")
        return r["model_list"]
    

class MultiAPIController:
    def __init__(self, apis: List[WebUIApi], sleep_interval=0.1):
        self.apis = apis
        self.sleep_interval = sleep_interval
        self.queue = defaultdict(dict) #txt2img, img2img, txt2img_task
        self.results = {}
        self._start_queue_thread()
        
    def generate_uuid(self):
        return str(uuid.uuid4())
        
    def wait_until_finished(self):
        import time
        while True:
            for api in self.apis:
                if api.is_ready():
                    return
            time.sleep(self.sleep_interval)
        
    def get_any_ready_api(self):
        for api in self.apis:
            if self.is_ready(api):
                return api
        return None
    
    def get_result(self, task_id):
        if task_id not in self.results.keys():
            raise RuntimeError(f"task id {task_id} is not found in results")
        return self.results[task_id]
    
    def _start_queue_thread(self):
        def queue_thread():
            import time
            while True:
                for api in self.apis:
                    if self.is_ready(api):
                        if len(self.queue["txt2img"]) > 0:
                            task_id, (args, kwargs) = self.queue["txt2img"].popitem()
                            self.results[task_id] = api.txt2img(*args, **kwargs)
                        elif len(self.queue["txt2img_task"]) > 0:
                            task_id, (args, kwargs) = self.queue["txt2img_task"].popitem()
                            self.results[task_id] = api.txt2img_task(*args, **kwargs)
                        elif len(self.queue["img2img"]) > 0:
                            task_id, (args, kwargs) = self.queue["img2img"].popitem()
                            self.results[task_id] = api.img2img(*args, **kwargs)
                time.sleep(self.sleep_interval)
        threading.Thread(target=queue_thread).start()
            
            
    def is_ready(self, api:WebUIApi):
        api_result = api.get_progress()
        progress = api_result["progress"]
        job_count = api_result["state"]["job_count"]
        return progress == 0.0 and job_count == 0
    
    def txt2img(self, *args, **kwargs):
        while True:
            self.wait_until_finished()
            ready_api = self.get_any_ready_api()
            if ready_api is None:
                continue
            return ready_api.txt2img(*args, **kwargs)
        
    def txt2img_task(self, *args, **kwargs):
        while True:
            self.wait_until_finished()
            ready_api = self.get_any_ready_api()
            if ready_api is None:
                continue
            return ready_api.txt2img_task(*args, **kwargs)
        
    def img2img(self, *args, **kwargs):
        while True:
            self.wait_until_finished()
            ready_api = self.get_any_ready_api()
            if ready_api is None:
                continue
            return ready_api.img2img(*args, **kwargs)
        
    def queue_txt2img(self, *args, **kwargs):
        self.queue["txt2img"][self.generate_uuid()] = (args, kwargs)
        
    def queue_txt2img_task(self, *args, **kwargs):
        self.queue["txt2img_task"][self.generate_uuid()] = (args, kwargs)
        
    def queue_img2img(self, *args, **kwargs):
        self.queue["img2img"][self.generate_uuid()] = (args, kwargs)
        