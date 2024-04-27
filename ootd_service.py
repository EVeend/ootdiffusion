from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing

from utils_ootd import get_mask_location

from PIL import Image

SCALE=2.0
STEP=20
SAMPLE=4
GPU_ID=0

async def predict_service(model_type, model_image, cloth_image, category, seed):

    openpose_model = OpenPose(GPU_ID)
    parsing_model = Parsing(GPU_ID)

    category_dict = ['upperbody', 'lowerbody', 'dress']
    category_dict_utils = ['upper_body', 'lower_body', 'dresses']

    if model_type == "hd":
        model = OOTDiffusionHD(GPU_ID)
    elif model_type == "dc":
        model = OOTDiffusionDC(GPU_ID)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")
    
    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")
    
    cloth_img = Image.open(cloth_image.file.read()).resize((768, 1024))
    model_img = Image.open(model_image.file.read()).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    masked_vton_img.save('./images_output/mask.jpg')
    
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=SAMPLE,
        num_steps=STEP,
        image_scale=SCALE,
        seed=seed,
    )

    image_idx = 0
    for image in images:
        print("IMAGE TYPE", type(image))
        image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
        image_idx += 1
