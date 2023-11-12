#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
import torch
import onnxruntime
import pathlib

from time import time

import ogi.globals
import ogi.metadata
import ogi.utilities as util
import ogi.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from ogi.face_util import extract_face_images
from ogi.ProcessEntry import ProcessEntry
from ogi.ProcessMgr import ProcessMgr
from ogi.ProcessOptions import ProcessOptions
from ogi.capturer import get_video_frame_total


clip_text = None

call_display_ui = None

process_mgr = None


if 'ROCMExecutionProvider' in ogi.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    ogi.globals.headless = False
    # Always enable all processors when using GUI
    if len(sys.argv) > 1:
        print('No CLI args supported - use Settings Tab instead')
    ogi.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in ogi.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in ogi.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage
    if ogi.globals.max_memory:
        memory = ogi.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = ogi.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def release_resources() -> None:
    import gc
    global process_mgr

    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None

    gc.collect()
    # if 'CUDAExecutionProvider' in ogi.globals.execution_providers and torch.cuda.is_available():
    #     with torch.cuda.device('cuda'):
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = util.resolve_relative_path('../models')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
    util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    util.conditional_download(download_directory_path, ['https://github.com/facefusion/facefusion-assets/releases/download/models/GPEN-BFR-512.onnx'])

    download_directory_path = util.resolve_relative_path('../models/CLIP')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if ogi.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(ogi.globals.source_path,  (False, 0))
        # ogi.globals.INPUT_FACES.append(faces[ogi.globals.source_face_index])
        # faces = extract_face_images(ogi.globals.target_path,  (False, util.has_image_extension(ogi.globals.target_path)))
        # ogi.globals.TARGET_FACES.append(faces[ogi.globals.target_face_index])
        # if 'face_enhancer' in ogi.globals.frame_processors:
        #     ogi.globals.selected_enhancer = 'GFPGAN'
       
    batch_process(None, False, None)


def get_processing_plugins(use_clip):
    processors = "faceswap"
    if use_clip:
        processors += ",mask_clip2seg"
    
    if ogi.globals.selected_enhancer == 'GFPGAN':
        processors += ",gfpgan"
    elif ogi.globals.selected_enhancer == 'Codeformer':
        processors += ",codeformer"
    elif ogi.globals.selected_enhancer == 'DMDNet':
        processors += ",dmdnet"
    elif ogi.globals.selected_enhancer == 'GPEN':
        processors += ",gpen"
    return processors


def live_swap(frame, swap_mode, use_clip, clip_text, selected_index = 0):
    global process_mgr

    if frame is None:
        return frame

    if process_mgr is None:
        process_mgr = ProcessMgr(None)
    
    options = ProcessOptions(get_processing_plugins(use_clip), ogi.globals.distance_threshold, ogi.globals.blend_ratio, swap_mode, selected_index, clip_text)
    process_mgr.initialize(ogi.globals.INPUT_FACESETS, ogi.globals.TARGET_FACES, options)
    newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def preview_mask(frame, clip_text):
    import numpy as np
    global process_mgr
    
    maskimage = np.zeros((frame.shape), np.uint8)
    if process_mgr is None:
        process_mgr = ProcessMgr(None)
    options = ProcessOptions("mask_clip2seg", ogi.globals.distance_threshold, ogi.globals.blend_ratio, "None", 0, clip_text)
    process_mgr.initialize(ogi.globals.INPUT_FACESETS, ogi.globals.TARGET_FACES, options)
    maskprocessor = next((x for x in process_mgr.processors if x.processorname == 'clip2seg'), None)
    return process_mgr.process_mask(maskprocessor, frame, maskimage)
    




def batch_process(files:list[ProcessEntry], use_clip, new_clip_text, use_new_method, progress) -> None:
    global clip_text, process_mgr

    ogi.globals.processing = True
    release_resources()
    limit_resources()

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        ogi.globals.execution_threads = 1

    imagefiles:list[ProcessEntry] = []
    videofiles:list[ProcessEntry] = []
           
    update_status('Sorting videos/images')


    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, ogi.globals.output_path, f'.{ogi.globals.CFG.output_image_format}')
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            destination = util.get_destfilename_from_path(fullname, ogi.globals.output_path, f'__temp.{ogi.globals.CFG.output_video_format}')
            f.finalname = destination
            videofiles.append(f)


    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    
    options = ProcessOptions(get_processing_plugins(use_clip), ogi.globals.distance_threshold, ogi.globals.blend_ratio, ogi.globals.face_swap_mode, 0, new_clip_text)
    process_mgr.initialize(ogi.globals.INPUT_FACESETS, ogi.globals.TARGET_FACES, options)

    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        origimages = []
        fakeimages = []
        for f in imagefiles:
            origimages.append(f.filename)
            fakeimages.append(f.finalname)

        process_mgr.run_batch(origimages, fakeimages, ogi.globals.execution_threads)
        origimages.clear()
        fakeimages.clear()

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not ogi.globals.processing:
                end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')
            start_processing = time()
            if ogi.globals.keep_frames or not use_new_method:
                util.create_temp(v.filename)
                update_status('Extracting frames...')
                ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                if not ogi.globals.processing:
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths, ogi.globals.execution_threads)
                if not ogi.globals.processing:
                    end_processing('Processing stopped!')
                    return
                if ogi.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)                                    
                
                ffmpeg.create_video(v.filename, f.finalname, fps)
                if not ogi.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ['gif']):
                    skip_audio = True
                else:
                    skip_audio = ogi.globals.skip_audio
                process_mgr.run_batch_inmem(v.filename, v.finalname, v.startframe, v.endframe, fps,ogi.globals.execution_threads, skip_audio)
                
            if not ogi.globals.processing:
                end_processing('Processing stopped!')
                return
            
            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ''
                if util.has_extension(v.filename, ['gif']):
                    gifname = util.get_destfilename_from_path(v.filename, ogi.globals.output_path, '.gif')
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    update_status('Creating final GIF')
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = ogi.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    if not skip_audio:
                        ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)
                update_status(f'\nProcessing {os.path.basename(destination)} took {time() - start_processing} secs')

            else:
                update_status(f'Failed processing {os.path.basename(v.finalname)}!')
    end_processing('Finished')


def end_processing(msg:str):
    update_status(msg)
    ogi.globals.target_folder_path = None
    release_resources()


def destroy() -> None:
    if ogi.globals.target_path:
        util.clean_temp(ogi.globals.target_path)
    release_resources()        
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    ogi.globals.CFG = Settings('config.yaml')
    ogi.globals.execution_threads = ogi.globals.CFG.max_threads
    ogi.globals.video_encoder = ogi.globals.CFG.output_video_codec
    ogi.globals.video_quality = ogi.globals.CFG.video_quality
    ogi.globals.max_memory = ogi.globals.CFG.memory_limit if ogi.globals.CFG.memory_limit > 0 else None
    main.run()
