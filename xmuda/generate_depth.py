from xmuda.depth_networks import ResnetEncoder, DepthDecoder
import torch
from xmuda.data.semantic_kitti.semantic_kitti_quan import SemanticKITTISCN
import os
from torch.utils.data.dataloader import DataLoader
from xmuda.data.semantic_kitti.collate import collate_fn
from xmuda.common.utils.torch_util import worker_init_fn
from torchvision import transforms, datasets
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
from xmuda.depth_networks.layers import disp_to_depth

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

encoder = ResnetEncoder(18, False)
depth_decoder = DepthDecoder(encoder.num_ch_enc)

encoder_path = "/home/docker_user/workspace/xmuda/weights/encoder.pth"
depth_decoder_path = "/home/docker_user/workspace/xmuda/weights/depth.pth"


preprocess_dir = '/datasets_local/datasets_acao/semantic_kitti_preprocess/preprocess'
semantic_kitti_dir = '/datasets_master/semantic_kitti'

# train_ds = SemanticKITTISCN(split=('train',),
#                             preprocess_dir=preprocess_dir,
#                             semantic_kitti_dir=semantic_kitti_dir,
#                             img_h=640,
#                             img_w=192,
#                             down_sample=1,
#                             #  merge_classes=False,
#                             noisy_rot=0.0,
#                             rot_z=0,
#                             transl=False,
#                             bottom_crop=None,
#                             fliplr=0,
#                             color_jitter=None,
#                             normalize_image=False
#                             )
# train_dataloader = DataLoader(
#             train_ds,
#             batch_size=4,
#             drop_last=True,
#             num_workers=4,
#             shuffle=True,
#             pin_memory=True,
#             worker_init_fn=worker_init_fn,
#             collate_fn=collate_fn
#         )

device = torch.device("cuda")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {
    k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()
# with torch.no_grad():
#     for data in train_dataloader:
#         input_color = data["img"].cuda()

#         #    if opt.post_process:
#         #         # Post-processed results require each image to have two forward passes
#         #         input_color = torch.cat(
#         #             (input_color, torch.flip(input_color, [3])), 0)
#         t = encoder(input_color)
#         # print(t.shape)
#         output = depth_decoder(t)
#         # print(output)
#         break
with torch.no_grad():
    # Load image and preprocess
    image_path = "/home/docker_user/workspace/xmuda/images/000005.png"
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)
    

    output_directory = "/home/docker_user/workspace/xmuda/images/depths"

    # Saving numpy file
    output_name = os.path.splitext(os.path.basename(image_path))[0]
    name_dest_npy = os.path.join(
        output_directory, "{}_depth.npy".format(output_name))
    scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
    pred_depth = 5.4 / scaled_disp
    pred_depth = pred_depth.cpu().numpy()
    print(pred_depth.shape, np.max(pred_depth), np.min(pred_depth))
    np.save(name_dest_npy, pred_depth)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[
        :, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    output_name = "test"
    name_dest_im = os.path.join(
        output_directory, "{}_disp.jpeg".format(output_name))
    im.save(name_dest_im)
