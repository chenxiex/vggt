from vggt.models.vggt import VGGT
import torch
from typing import List
from pathlib import Path
from evaluation.memory_profiler import PredictionMemoryProfiler
from vggt.utils.load_fn import load_and_preprocess_images
import re
import numpy as np
from PIL import Image
import torch.nn.functional as F
import open3d as o3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    dtype = torch.float32


def load_model(model_path) -> VGGT:
    model = VGGT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    return model


def predict(images_path: List[Path], model: VGGT):
    profiler = PredictionMemoryProfiler(model, device=device, dtype=dtype)
    profiler.record_snapshot("before_predict")

    sampled_image_names = [str(p) for p in images_path]
    images = load_and_preprocess_images(sampled_image_names).to(device)
    profiler.record_input_metadata(images, images_path)
    profiler.record_snapshot("after_input_to_device")

    profiler.install_hooks()

    try:
        with torch.no_grad():
            profiler.record_snapshot("before_model_forward")
            with torch.amp.autocast('cuda', dtype=dtype): # pyright: ignore[reportPrivateImportUsage]
                # Predict attributes including cameras, depth maps, and point maps
                predictions = model(images)
            profiler.record_snapshot("after_model_forward")
    except Exception as exc:
        profiler.record_error(exc)
        raise
    finally:
        profiler.remove_hooks()
        profiler.finalize()
    return predictions


def read_pfm(filename):
    file = open(filename, 'rb')  # 1. 以二进制读模式打开文件
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()  # 2. 读取第一行头部信息
    if header == 'PF':
        color = True  # 彩色图像
    elif header == 'Pf':
        color = False  # 灰度图像
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                         file.readline().decode('utf-8'))  # 3. 读取第二行，解析宽度和高度
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())  # 4. 读取第三行，解析缩放因子和字节序
    if scale < 0:  # little-endian 小端序
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian 大端序

    # 5. 读取剩余的二进制数据
    # np.fromfile 从文件中读取数据，并指定字节序和数据类型 ('f' 代表 float32)
    data = np.fromfile(file, endian + 'f')
    # 6. 根据是否为彩色图确定数据形状
    shape = (height, width, 3) if color else (height, width)

    # 7. 重塑数据并翻转行
    # PFM 文件的行序是颠倒的 (从下到上存储)，所以需要使用 np.flipud 进行翻转
    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()  # 8. 关闭文件
    return data, scale  # 9. 返回解析后的数据和缩放因子

def upsample_image(image: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    """
    Args:
        image: 源图像
        target_w: 目标宽度
        target_h: 目标高度
    Returns:
        上采样后的图像
    """

    image = image.squeeze()

    # 2. 转成 numpy（float32）
    image_np = image.detach().cpu().numpy().astype(np.float32)

    # 3. 用 PIL BICUBIC 上采样
    img = Image.fromarray(image_np)
    img_up_np = np.array(
        img.resize((target_h, target_w), Image.Resampling.BICUBIC)
    )

    # 4. 转回 torch Tensor
    img_up = torch.from_numpy(img_up_np).float()

    return img_up


def upsample_images(images: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    upsampled_images = []

    for d in images:
        d = upsample_image(d, target_w, target_h)
        upsampled_images.append(d)

    upsampled_images = torch.stack(upsampled_images, dim=0)
    return upsampled_images


def generate_points_from_depth(depth, proj):
    '''
    :param depth: (B, 1, H, W)
    :param proj: (B, 4, 4)
    :return: point_cloud (B, 3, H, W)
    '''
    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    inv_proj = torch.inverse(proj)

    rot = inv_proj[:, :3, :3]  # [B,3,3]
    trans = inv_proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    # [u,v,1]
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    # (RK)^{-1}*[u,v,1]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    # (RK)^{-1}*[u,v,1]*d
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
    # (RK)^{-1}*[u,v,1]*d+t
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
    proj_xyz = proj_xyz.view(batch, 3, height, width)

    return proj_xyz


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    '''
    该函数将src_fea从src_proj投影到ref_proj。首先利用src_proj和ref_proj算出depth_values在src上的投影，然后根据这个投影的坐标对src_fea进行采样。
    Args:
        src_fea: (B, C, H, W)
        src_proj: (B, 4, 4)
        ref_proj: (B, 4, 4)
        depth_values: (B, H, W)
    Returns:
        warped_src_fea: (B, C, H, W)
    '''
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(
            2) * depth_values.view(-1, 1, 1, height*width)  # [B, 3, 1, H*W]

        proj_xyz = rot_depth_xyz + \
            trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / \
            proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        # [B, Ndepth, H*W, 2]
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch,  height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    return warped_src_fea


def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0])**2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1])**2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2])**2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist


def write_ply(file: Path, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)


def extract_points(pc, mask, rgb):
    pc = pc.cpu()
    mask = mask.cpu()
    rgb = rgb.cpu()

    pc = pc.numpy()
    mask = mask.numpy()
    rgb = rgb.numpy()

    mask = np.reshape(mask, (-1,))
    pc = np.reshape(pc, (-1, 3))
    rgb = np.reshape(rgb, (-1, 3))

    points = pc[np.where(mask)]
    colors = rgb[np.where(mask)]

    points_with_color = np.concatenate([points, colors], axis=1)

    return points_with_color


def open3d_filter(depths: torch.Tensor, projs: torch.Tensor, rgbs: torch.Tensor, dist_thresh: float = 1.0, batch_size: int = 20, num_consist: int = 4):
    with torch.no_grad():
        tot_frame = depths.shape[0]
        height, width = depths.shape[2], depths.shape[3]
        points = []

        for i in range(tot_frame):
            pc_buff = torch.zeros((3, height, width),
                                  device=depths.device, dtype=depths.dtype)
            val_cnt = torch.zeros((1, height, width),
                                  device=depths.device, dtype=depths.dtype)
            j = 0

            while True:
                ref_pc, pcs, dist = filter_depth(
                    ref_depth=depths[i:i+1],
                    src_depths=depths[j:min(j+batch_size, tot_frame)],
                    ref_proj=projs[i:i+1],
                    src_projs=projs[j:min(j+batch_size, tot_frame)]
                )

                depth_mask = (dist < dist_thresh).float()

                masks = depth_mask

                masked_pc = pcs * masks
                pc_buff += masked_pc.sum(dim=0, keepdim=False)
                val_cnt += masks.sum(dim=0, keepdim=False)

                j += batch_size
                if j >= tot_frame:
                    break

            final_mask = (val_cnt >= num_consist).squeeze(0)
            avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)

            final_pc = extract_points(avg_points, final_mask, rgbs[i])
            points.append(final_pc)

        points = np.concatenate(points, axis=0)
        return points