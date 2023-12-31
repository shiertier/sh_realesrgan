import argparse
from PIL import Image
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np

def from_model_name_determine_model(model_name="RealESRGAN_x4plus_anime_6B"):
    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://hub.yzuu.cf/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://hub.yzuu.cf/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://hub.yzuu.cf/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://hub.yzuu.cf/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://hub.yzuu.cf/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://hub.yzuu.cf/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://hub.yzuu.cf/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    return model, netscale, file_url

def get_model_path(model_path, model_name, file_url):
    if model_path is None:
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            #ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            ROOT_DIR = "/gemini/code"
            for url in file_url:
                model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    return model_path

def use_dni_control_denoise(model_path, denoise_strength):
    wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    model_path = [model_path, wdn_model_path]
    dni_weight = [denoise_strength, 1 - denoise_strength]
    return model_path, dni_weight

#model_name, model_path=None, denoise_strength=1, tile=0,tile_pad,pre_pad,half,gpu_id,face_enhance,upscale,outscale,ext,suffix,input_path
def realesrgan_inference(input_path="input", model_name="RealESRGAN_x4plus_anime_6B", output_path="result", denoise_strength=0.25, outscale=4, model_path=None, suffix=None, tile=0, tile_pad=10,pre_pad=10,face_enhance=False,fp32=False,ext="auto",gpu_id=None):
    """Inference demo for Real-ESRGAN.
    Args:
        input_path (str): Path to the input image or folder.
        model_name (str): Model name: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3 | realesr-general-x4v3.
        output_path (str): Path to the output folder.
        denoise_strength (float): Denoise strength. 0 means weak denoising (preserve noise), 1 means strong denoising capability. Only used for realesr-general-x4v3 model.
        outscale (float): Scale factor for final upsampling of the image.
        model_path (str, optional): [Optional] Path to the model. Usually you don't need to specify it. Defaults to None.
        suffix (str, optional): Suffix for the enhanced image. Defaults to None.
        tile (int, optional): Tile size during testing, 0 means no tiling. Defaults to 0.
        tile_pad (int, optional): Padding for tiling. Defaults to 10.
        pre_pad (int, optional): Pre-padding size on each border. Defaults to 0.
        face_enhance (bool, optional): Use GFPGAN to enhance faces. Defaults to False.
        fp32 (bool, optional): Use fp32 for inference. Defaults to False.
        ext (str, optional): Output image format. "auto" means same as input format. Defaults to "auto".
        gpu_id (int, optional): GPU device id. Defaults to None.    
    """
    # Check input and output paths
    if not os.path.exists(input_path):
        raise ValueError(f'Input path {input_path} does not exist.')
    
    os.makedirs(output_path, exist_ok=True)
    
    # 根据模型名称确定模型
    model, netscale, file_url = from_model_name_determine_model(model_name)

    # 确定模型路径
    model_path = get_model_path(model_path, model_name, file_url)

    # 使用DNI控制去噪强度
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        model_path, dni_weight = use_dni_control_denoise(model_path, denoise_strength)

    # restorer
    half = not fp32
    upsampler = RealESRGANer(netscale,model_path,dni_weight,model,tile,tile_pad,pre_pad,half,gpu_id)

    if face_enhance:  # 面部加强
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://hub.yzuu.cf/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        paths = sorted(glob.glob(os.path.join(input_path, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = Image.open(path)
        if img.mode == 'RGBA':
            img_mode = 'RGBA'
        else:
            img_mode = None
        try:
            img = np.array(img)
            if face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if ext == 'auto':
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if suffix is None:
                save_path = os.path.join(output_path, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(output_path, f'{imgname}_{suffix}.{extension}')
            image = Image.fromarray(output)
            image.save(save_path)
        del img, output
        if face_enhance:
            del _, _
    # 释放模型资源
    del upsampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='输入图像或文件夹')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus_anime_6B',
        help=('模型名称： RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
            'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='输出文件夹')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.25,
        help=('去噪强度。0表示弱去噪（保留噪声），1表示强去噪能力。'
            '仅用于realesr-general-x4v3模型'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='图像最终上采样的比例')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[可选]模型路径。通常，您不需要指定它')
    parser.add_argument('--suffix', type=str, default=None, help='恢复图像的后缀')
    parser.add_argument('-t', '--tile', type=int, default=0, help='测试时的平铺大小，0表示不进行平铺')
    parser.add_argument('--tile_pad', type=int, default=10, help='平铺填充')
    parser.add_argument('--pre_pad', type=int, default=0, help='每个边界的预填充大小')
    parser.add_argument('--face_enhance', action='store_true', help='使用GFPGAN增强面部')
    parser.add_argument(
        '--fp32', action='store_true', help='推理时使用fp32精度。默认：fp16（半精度）')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='图像扩展名。选项：auto | jpg | png，auto表示使用与输入相同的扩展名')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='要使用的gpu设备（默认为None），可以是0，1，2用于多gpu')

    args = parser.parse_args()

    input_path=args.input
    model_name = args.model_name.split('.')[0]
    output_path=args.output
    denoise_strength = args.denoise_strength
    outscale=args.outscale
    model_path = args.model_path
    suffix = args.suffix
    tile = args.tile
    tile_pad = args.tile_pad
    pre_pad = args.pre_pad
    face_enhance = args.face_enhance
    fp32=args.fp32
    ext = args.ext
    gpu_id = args.gpu_id

    a(input_path, model_name, output_path, denoise_strength, outscale, model_path, suffix, tile, tile_pad,pre_pad,face_enhance,fp32,ext,gpu_id)
    
if __name__ == '__main__':
    main()
