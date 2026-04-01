import os
import sys
import logging
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

from utils.data_loading import BasicDataset
from utils.dataset_process import compute_mean_std
from utils.path_hyperparameter import ph

# ✅ 用你现在这份模型
from models.Modelsxiao import DPCD


# =========================
# 配置
# =========================
DATASET_NAME = "LEVIRRRRR"
SAVE_DIR = r"D:\projectsonline\offical-SGSLN-main\log_feature\heatmaps_multistage"
LOAD_CHECKPOINT = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_JET_R = False     # False=jet(蓝底红黄热点)；True=jet_r(反色)
SMOOTH_SIGMA = 0.8    # 0/None 不平滑
UPSAMPLE_TO_INPUT = True

# ✅ 适配你模型：真实模块名
HOOK_MODULES = [
    "en_block2", "en_block3", "en_block4", "en_block5",
    "masam",
    "csfb_att2", "csfb_att3", "csfb_att4", "csfb_att5",
    "de_block1", "de_block2", "de_block3",
    "dpfa1", "dpfa2", "dpfa3", "dpfa4",
    "change_block4", "change_block3", "change_block2",
    "upsample_x2",
    "conv_out_change",
]
# =========================


def jet_colormap(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def jet_r_colormap(x01: np.ndarray) -> np.ndarray:
    return jet_colormap(1.0 - np.clip(x01, 0.0, 1.0))


def maybe_smooth(hm01: np.ndarray, sigma: float):
    if sigma is None or sigma <= 0:
        return hm01
    try:
        import cv2
        hm = hm01.astype(np.float32)
        hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return np.clip(hm, 0.0, 1.0)
    except Exception:
        return hm01


def load_model_weights(net: torch.nn.Module, ckpt_path: str, device: torch.device, load_checkpoint: bool):
    assert ckpt_path, "ph.load 不能为空：请在 utils/path_hyperparameter.py 里设置 ph.load 为权重路径"
    logging.info(f"Loading model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if load_checkpoint and isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt)
    else:
        state_dict = ckpt

    net.load_state_dict(state_dict, strict=True)
    logging.info("Model weights loaded OK.")


# -------------------------
# 关键：支持“同一模块多次被调用” -> 存 list
# -------------------------
class MultiCallFeatureCatcher:
    """
    features[module_name] = [out_call0, out_call1, ...]
    out 可以是 Tensor / tuple / list（原样存），后面保存时再展开
    """
    def __init__(self, net: torch.nn.Module, module_names):
        self.net = net
        self.module_names = set(module_names)
        self.features = defaultdict(list)
        self.handles = []
        self._register()

    def _register(self):
        name2module = dict(self.net.named_modules())
        missing = [n for n in self.module_names if n not in name2module]
        if missing:
            logging.warning("Some hook modules not found:")
            for n in missing:
                logging.warning(f"  - {n}")
            logging.warning("Please check model.named_modules() names.")

        for n in self.module_names:
            if n not in name2module:
                continue
            m = name2module[n]
            h = m.register_forward_hook(self._make_hook(n))
            self.handles.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            self.features[name].append(out)
        return hook

    def clear(self):
        self.features.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def feature_to_heat01(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: [B,C,H,W] / [B,1,H,W] / [B,H,W]
    -> [B,1,H,W] in [0,1]
    """
    if feat.dim() == 3:
        heat = feat.unsqueeze(1)
    elif feat.dim() == 4:
        if feat.size(1) == 1:
            heat = feat
        else:
            heat = feat.abs().mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported feat shape: {feat.shape}")

    b = heat.size(0)
    flat = heat.view(b, -1)
    mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
    heat01 = (heat - mn) / (mx - mn + 1e-6)
    return heat01.clamp(0, 1)


def save_heat01_batch(heat01_b1hw: torch.Tensor, names, out_dir: str,
                      prefix: str, use_jet_r: bool, smooth_sigma: float):
    os.makedirs(out_dir, exist_ok=True)
    arr = heat01_b1hw.detach().cpu().numpy()  # [B,1,H,W]
    b = arr.shape[0]
    for i in range(b):
        hm = arr[i, 0]
        hm = maybe_smooth(hm, smooth_sigma)

        img_name = str(names[i])
        img_base = os.path.basename(img_name)
        img_id = os.path.splitext(img_base)[0]

        rgb = jet_r_colormap(hm) if use_jet_r else jet_colormap(hm)
        Image.fromarray(rgb, mode="RGB").save(os.path.join(out_dir, f"{img_id}__{prefix}.png"))


def maybe_upsample(heat01: torch.Tensor, input_hw):
    if input_hw is None:
        return heat01
    return F.interpolate(heat01, size=input_hw, mode="bilinear", align_corners=False)


def save_caught_features(caught: dict, names, base_dir: str, input_hw=None,
                         use_jet_r: bool = False, smooth_sigma: float = 0.0):
    """
    针对你模型的输出结构做展开：
    - en_block2~5: 每个模块 call0=t1, call1=t2
    - masam: call0=t1, call1=t2；每个 call 输出 (s2,s3,s4,s5)
    - csfb_attk: call0对应尺度k；输出 (t1_enhanced, t2_enhanced)
    - de_block1~3: 每个模块 call0=t1, call1=t2
    - dpfa1~4: 一次输出 tensor
    - change_block4~2: 一次输出 tensor
    - upsample_x2: 一次输出 tensor
    - conv_out_change: 一次输出 tensor（logits）
    """
    os.makedirs(base_dir, exist_ok=True)

    def save_tensor(stage, tensor, suffix):
        heat01 = feature_to_heat01(tensor.detach())
        heat01 = maybe_upsample(heat01, input_hw)
        out_dir = os.path.join(base_dir, stage)
        save_heat01_batch(heat01, names, out_dir, prefix=suffix, use_jet_r=use_jet_r, smooth_sigma=smooth_sigma)

    # 统一映射：call0/call1 -> t1/t2（你 forward 就是这个顺序）
    call_tag = {0: "t1", 1: "t2"}

    for module_name, outs in caught.items():
        for ci, out in enumerate(outs):
            tag = call_tag.get(ci, f"call{ci}")

            # masam: out 是 (s2,s3,s4,s5)
            if module_name == "masam" and isinstance(out, (tuple, list)) and len(out) == 4:
                s2, s3, s4, s5 = out
                save_tensor("masam_s2", s2, f"{tag}")
                save_tensor("masam_s3", s3, f"{tag}")
                save_tensor("masam_s4", s4, f"{tag}")
                save_tensor("masam_s5", s5, f"{tag}")
                continue

            # csfb_att*: out 是 (t1_enhanced, t2_enhanced)
            if module_name.startswith("csfb_att") and isinstance(out, (tuple, list)) and len(out) == 2:
                t1e, t2e = out
                stage = module_name  # 例如 csfb_att2
                save_tensor(stage, t1e, "t1_enhanced")
                save_tensor(stage, t2e, "t2_enhanced")
                continue

            # 普通 tensor 输出
            if torch.is_tensor(out):
                stage = module_name
                # en_block/de_block 是双分支多次调用，命名带 t1/t2
                if module_name.startswith("en_block") or module_name.startswith("de_block"):
                    save_tensor(stage, out, tag)
                else:
                    # dpfa/change/upsample/conv_out_change 一般只调用一次
                    save_tensor(stage, out, "out")
                continue

            # 其它结构（极少）
            logging.warning(f"[save_caught_features] skip {module_name} call{ci}, type={type(out)}")


def save_final_prob_heatmap(prob_b1hw: torch.Tensor, names, save_dir: str,
                            use_jet_r: bool = False, smooth_sigma: float = 0.0):
    os.makedirs(save_dir, exist_ok=True)
    arr = prob_b1hw.detach().cpu().numpy()
    b = arr.shape[0]
    for i in range(b):
        hm = maybe_smooth(arr[i, 0], smooth_sigma)

        img_name = str(names[i])
        img_base = os.path.basename(img_name)
        img_id = os.path.splitext(img_base)[0]

        rgb = jet_r_colormap(hm) if use_jet_r else jet_colormap(hm)
        Image.fromarray(rgb, mode="RGB").save(os.path.join(save_dir, f"{img_id}_final_prob.png"))


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {DEVICE}")

    # mean/std
    t1_mean, t1_std = compute_mean_std(images_dir=f'./{DATASET_NAME}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'./{DATASET_NAME}/train/t2/')
    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)

    test_dataset = BasicDataset(
        t1_images_dir=f'./{DATASET_NAME}/test/t1/',
        t2_images_dir=f'./{DATASET_NAME}/test/t2/',
        labels_dir=f'./{DATASET_NAME}/test/label/',
        train=False,
        **dataset_args
    )

    loader_args = dict(num_workers=8, prefetch_factor=5, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=ph.batch_size * ph.inference_ratio,
        **loader_args
    )

    net = DPCD().to(device=DEVICE)
    load_model_weights(net, ph.load, DEVICE, load_checkpoint=LOAD_CHECKPOINT)
    net.eval()

    catcher = MultiCallFeatureCatcher(net, HOOK_MODULES)

    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=DEVICE),
        'precision': Precision().to(device=DEVICE),
        'recall': Recall().to(device=DEVICE),
        'f1score': F1Score().to(device=DEVICE)
    })

    stages_dir = os.path.join(SAVE_DIR, "stages")
    final_dir = os.path.join(SAVE_DIR, "final")
    os.makedirs(stages_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    logging.info(f"Stage heatmaps -> {stages_dir}")
    logging.info(f"Final heatmaps  -> {final_dir}")

    with torch.no_grad():
        for batch_img1, batch_img2, labels, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(DEVICE)
            batch_img2 = batch_img2.float().to(DEVICE)
            labels = labels.float().to(DEVICE)

            input_hw = (batch_img1.shape[-2], batch_img1.shape[-1]) if UPSAMPLE_TO_INPUT else None

            catcher.clear()

            # ✅ 用 log=True 不影响 hook，同时你模型内部也会 log_feature（若你开了）
            out = net(batch_img1, batch_img2, log=False, img_name=name, label=labels)

            # out = (change_out, seg_out1, seg_out2)
            change_out = out[0]
            final_prob = torch.sigmoid(change_out)

            # 1) 保存所有阶段 feature 热力图
            save_caught_features(
                catcher.features,
                name,
                stages_dir,
                input_hw=input_hw,
                use_jet_r=USE_JET_R,
                smooth_sigma=SMOOTH_SIGMA
            )

            # 2) 保存 final prob 热力图（最直观）
            save_final_prob_heatmap(final_prob, name, final_dir, use_jet_r=USE_JET_R, smooth_sigma=SMOOTH_SIGMA)

            # 3) 指标（按你的原习惯）
            labels_i = labels.int().unsqueeze(1)
            metric_collection.update(final_prob.float(), labels_i)

            del batch_img1, batch_img2, labels

    test_metrics = metric_collection.compute()
    print(f"Metrics on all data: {test_metrics}")
    metric_collection.reset()
    print("over")

    catcher.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
