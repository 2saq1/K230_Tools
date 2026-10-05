#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K230 模型工厂  —  从图片数据集一键生成 K230 可用的 kmodel

整合流程：数据校验 → 切分 → 训练 → 导出ONNX → 转kmodel → 输出配置片段

用法：
  双击桌面「K230 模型工厂」图标（GUI 模式）
  或命令行： python k230_model_factory.py --run config.json   （无界面流水线模式）
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import subprocess
import traceback
from datetime import datetime

# ------------------------------------------------------------------ 常量
HERE = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(HERE, "tools")
TO_KMODEL = os.path.join(TOOL_DIR, "to_kmodel.py")

# 便携包根目录（本文件位于 <包>\apps\factory\ 下，往上两级）
PKG_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
WS = os.path.join(PKG_ROOT, "workspace")

# 用 pythonw.exe 启动 GUI 时，子进程必须换成 python.exe 才能正常拿到输出
PY_EXE = sys.executable
if os.path.basename(PY_EXE).lower() == "pythonw.exe":
    _alt = os.path.join(os.path.dirname(PY_EXE), "python.exe")
    if os.path.isfile(_alt):
        PY_EXE = _alt

# 子进程不弹黑窗
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

MODEL_CHOICES = ["yolo11n", "yolo11s", "yolov8n", "yolov8s"]
IMGSZ_CHOICES = [320, 416, 640]
SPLIT_RATES = (0.7, 0.2, 0.1)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def log(msg, stage="", pct=None):
    """输出一行进度。GUI 通过 stdout 解析这些 JSON 行。"""
    rec = {"t": datetime.now().strftime("%H:%M:%S"), "stage": stage, "msg": str(msg)}
    if pct is not None:
        rec["pct"] = pct
    print("@@" + json.dumps(rec, ensure_ascii=False), flush=True)


# ------------------------------------------------------------------ 数据集
def describe_dir(path):
    """给报错用的目录说明：列出前几项，让用户看清自己选到哪一层了。"""
    if not os.path.isdir(path):
        return "    （这个路径不存在）"
    subs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    lines = []
    lines.append("    子文件夹: %s" % ("、".join(subs[:8]) if subs else "（没有）"))
    lines.append("    文件数  : %d" % len(files))
    if files:
        lines.append("    文件名例: %s" % "、".join(sorted(files)[:4]))
    return "\n".join(lines)


def inspect_dataset(src_dir):
    """看看这个目录是不是 YOLO 格式（images/ + labels/）。"""
    img_dir = os.path.join(src_dir, "images")
    lbl_dir = os.path.join(src_dir, "labels")
    if not os.path.isdir(img_dir):
        base = os.path.basename(os.path.normpath(src_dir)).lower()
        # 允许直接给 images 目录，或者 labels 目录（自动回退到上一层）
        if base in ("images", "labels"):
            parent = os.path.dirname(os.path.normpath(src_dir))
            img_dir = os.path.join(parent, "images")
            lbl_dir = os.path.join(parent, "labels")
        else:
            return None
    if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
        return None

    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXT)]
    pairs, orphan = [], []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        lbl = stem + ".txt"
        if os.path.isfile(os.path.join(lbl_dir, lbl)):
            pairs.append((f, lbl))
        else:
            orphan.append(f)

    # 类别分布：扫描全部配对标签，统计每个 class id 的框数
    cls_counts, bad_lines, boxes = {}, [], 0
    for _, lbl in pairs:
        try:
            with open(os.path.join(lbl_dir, lbl), "r", encoding="utf-8", errors="ignore") as fh:
                for ln, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cid = int(float(line.split()[0]))
                    except Exception:
                        bad_lines.append("%s 第%d行" % (lbl, ln))
                        continue
                    cls_counts[cid] = cls_counts.get(cid, 0) + 1
                    boxes += 1
        except Exception:
            pass

    max_cls = max(cls_counts) if cls_counts else -1
    return {
        "img_dir": img_dir,
        "lbl_dir": lbl_dir,
        "images": len(imgs),
        "pairs": len(pairs),
        "orphan": len(orphan),
        "nc": max_cls + 1,
        "cls_counts": cls_counts,
        "boxes": boxes,
        "bad_lines": bad_lines,
    }


def split_dataset(src_dir, work_dir, info, seed=0):
    """把 images/labels 按 7:2:1 切分到 work/dataset/{train,valid,test}。"""
    dst_root = os.path.join(work_dir, "dataset")
    if os.path.isdir(dst_root):
        shutil.rmtree(dst_root)
    for sp in ("train", "valid", "test"):
        os.makedirs(os.path.join(dst_root, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst_root, sp, "labels"), exist_ok=True)

    imgs = [f for f in os.listdir(info["img_dir"]) if f.lower().endswith(IMAGE_EXT)]
    imgs.sort()
    random.Random(seed).shuffle(imgs)
    files = [f for f in imgs if os.path.isfile(os.path.join(info["lbl_dir"], os.path.splitext(f)[0] + ".txt"))]

    n = len(files)
    n_train = int(SPLIT_RATES[0] * n)
    n_val = n_train + int(SPLIT_RATES[1] * n)
    buckets = {"train": files[:n_train], "valid": files[n_train:n_val], "test": files[n_val:]}

    for sp, lst in buckets.items():
        for f in lst:
            shutil.copy(os.path.join(info["img_dir"], f), os.path.join(dst_root, sp, "images", f))
            shutil.copy(os.path.join(info["lbl_dir"], os.path.splitext(f)[0] + ".txt"),
                        os.path.join(dst_root, sp, "labels", os.path.splitext(f)[0] + ".txt"))
        log("%s: %d 张" % (sp, len(lst)), "split")

    return dst_root, {k: len(v) for k, v in buckets.items()}


def write_yaml(work_dir, dataset_root, names):
    p = os.path.join(work_dir, "data.yaml")
    lines = [
        "train: " + os.path.join(dataset_root, "train", "images").replace("\\", "/"),
        "val: " + os.path.join(dataset_root, "valid", "images").replace("\\", "/"),
        "test: " + os.path.join(dataset_root, "test", "images").replace("\\", "/"),
        "",
        "nc: %d" % len(names),
        "names: " + json.dumps(list(names), ensure_ascii=False),
        "",
    ]
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return p


def make_calib(dataset_root, work_dir, count=30):
    """量化校准集：从训练集里挑若干张。"""
    calib = os.path.join(work_dir, "calib")
    os.makedirs(calib, exist_ok=True)
    for f in os.listdir(calib):
        try:
            os.remove(os.path.join(calib, f))
        except Exception:
            pass
    src = os.path.join(dataset_root, "train", "images")
    imgs = [f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXT)][:count]
    for f in imgs:
        shutil.copy(os.path.join(src, f), os.path.join(calib, f))
    return calib, len(imgs)


# ------------------------------------------------------------------ 预训练权重
def find_weights(weights_root, model_name):
    """在 weights\\<family>\\<model>.pt 里找预训练权重。"""
    fam = None
    if model_name.startswith("yolov8"):
        fam = "yolov8"
    elif model_name.startswith("yolo11"):
        fam = "yolov11"
    elif model_name.startswith("yolov12"):
        fam = "yolov12"
    cands = []
    if fam:
        cands.append(os.path.join(weights_root, fam, model_name + ".pt"))
    cands.append(os.path.join(weights_root, model_name + ".pt"))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


# ------------------------------------------------------------------ 流水线
def run_pipeline(cfg):
    t0 = time.time()
    work = cfg["work_dir"]
    os.makedirs(work, exist_ok=True)

    # ---- 1 校验
    log("检查数据集 ...", "check")
    info = inspect_dataset(cfg["dataset_dir"])
    if not info:
        raise RuntimeError(
            "数据集格式不对，找不到 images 和 labels 两个文件夹。\n\n"
            "    你选的是: %s\n"
            "%s\n\n"
            "    要选「同时装着 images 和 labels 的那一层目录」，\n"
            "    也就是 images 和 labels 的【上一级】，例如：\n"
            "        C:\\Users\\SONG\\Desktop\\data\n"
            "    而不是里面的 images 或 labels。"
            % (cfg["dataset_dir"], describe_dir(cfg["dataset_dir"]))
        )
    log("图片 %d 张，其中 %d 张有标签，未配对 %d 张；标签里出现的最大类别号 = %d"
        % (info["images"], info["pairs"], info["orphan"], info["nc"]), "check")
    if info["pairs"] < 20:
        raise RuntimeError("有效样本只有 %d 张，太少了，至少准备 20 张以上" % info["pairs"])
    if info["orphan"]:
        log("提示：%d 张图没有对应标签，会被自动跳过" % info["orphan"], "check")

    names = cfg["names"]
    if not names:
        raise RuntimeError("没有填写类别名")

    cc = info["cls_counts"]
    log("共 %d 个标注框，类别分布：" % info["boxes"], "check")
    for cid in sorted(cc):
        nm = names[cid] if cid < len(names) else "——— 没有对应类别名"
        log("    类别号 %-4d -> %-24s %5d 个框" % (cid, nm, cc[cid]), "check")

    if info["bad_lines"]:
        log("注意：%d 行标签格式不合法（第一列不是数字），已忽略。例如：%s"
            % (len(info["bad_lines"]), "; ".join(info["bad_lines"][:3])), "check")

    # 硬校验：标签里的类别号必须都能对上你填的类别名，否则训练结果一定是错的
    over = [c for c in sorted(cc) if c >= len(names)]
    if over:
        raise RuntimeError(
            "类别名和标签对不上，已中止（继续下去训练出来的模型是错的）：\n\n"
            "    标签里出现类别号：%s\n"
            "    你填的类别名只有 %d 个：%s\n\n"
            "    类别号必须落在 0 ~ %d 之间。\n"
            "    请按标注时的顺序把类别名补齐（用英文逗号分隔），\n"
            "    或者换一个只有这几类的数据集。"
            % ("、".join(str(c) for c in over), len(names),
               "、".join(names), len(names) - 1)
        )

    absent = [i for i in range(len(names)) if cc.get(i, 0) == 0]
    if absent:
        log("警告：类别「%s」在标签里一个样本都没有，模型学不会它。"
            % "」「".join(names[i] for i in absent), "check")

    # ---- 2 切分
    log("切分数据集 7:2:1 ...", "split")
    ds_root, counts = split_dataset(cfg["dataset_dir"], work, info)
    yaml_path = write_yaml(work, ds_root, names)
    log("已生成 data.yaml", "split")

    # ---- 3 训练
    log("开始训练 %s @ %d ..." % (cfg["model"], cfg["imgsz"]), "train", 0)
    from ultralytics import YOLO
    w = find_weights(cfg["weights_root"], cfg["model"])
    if w:
        log("预训练权重：%s" % w, "train")
    else:
        log("没找到本地权重 %s.pt，ultralytics 会尝试联网下载" % cfg["model"], "train")
        w = cfg["model"] + ".pt"

    model = YOLO(w)
    train_kwargs = dict(
        data=yaml_path,
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        workers=int(cfg.get("workers", 4)),
        project=work,
        name="train",
        exist_ok=True,
        verbose=True,
    )
    try:
        model.train(**train_kwargs)
    except Exception as e:
        # 别的电脑上没显卡 / 驱动不匹配时自动退回 CPU
        if "cuda" in str(e).lower():
            log("CUDA 不可用（%s）... 自动改用 CPU 训练，会慢很多，请耐心等" % str(e)[:160], "train")
            train_kwargs["device"] = "cpu"
            model = YOLO(w)
            model.train(**train_kwargs)
        else:
            raise
    best_pt = os.path.join(work, "train", "weights", "best.pt")
    if not os.path.isfile(best_pt):
        raise RuntimeError("训练结束但找不到 best.pt：%s" % best_pt)
    log("训练完成：%s" % best_pt, "train", 100)

    # ---- 4 导出 ONNX
    log("导出 ONNX ...", "export", 0)
    y = YOLO(best_pt)
    onnx_path = y.export(format="onnx", imgsz=int(cfg["imgsz"]), opset=12,
                         simplify=False, dynamic=False)   # to_kmodel.py 内部会做 onnxsim，这里不用重复
    if not os.path.isfile(onnx_path):
        cand = os.path.splitext(best_pt)[0] + ".onnx"
        if os.path.isfile(cand):
            onnx_path = cand
    log("ONNX: %s (%.2f MB)" % (onnx_path, os.path.getsize(onnx_path) / 1048576.0), "export", 100)

    # ---- 5 转 kmodel
    log("转换 kmodel（nncase 2.11，PTQ 量化）...", "convert", 0)
    calib, ncal = make_calib(ds_root, work)
    log("校准集：%d 张" % ncal, "convert")
    cmd = [
        PY_EXE, TO_KMODEL,
        "--target", "k230",
        "--model", onnx_path,
        "--dataset", calib,
        "--input_width", str(cfg["imgsz"]),
        "--input_height", str(cfg["imgsz"]),
        "--ptq_option", "0",
    ]
    log("$ " + " ".join('"%s"' % c if " " in c else c for c in cmd), "convert")
    # nncase 需要 .NET：优先用包内自带（便携模式），未设置时才退回系统
    env = dict(os.environ)
    if not env.get("DOTNET_ROOT"):
        dr = os.path.join(PKG_ROOT, "dotnet")
        if os.path.isdir(dr):
            env["DOTNET_ROOT"] = dr
    proc = subprocess.Popen(cmd, cwd=TOOL_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", creationflags=_NO_WINDOW, env=env)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line, "convert")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("kmodel 转换失败，退出码 %d" % proc.returncode)

    kmodel_src = os.path.splitext(onnx_path)[0] + ".kmodel"
    if not os.path.isfile(kmodel_src):
        raise RuntimeError("转换结束但没找到 kmodel：%s" % kmodel_src)
    log("kmodel: %s (%.2f MB)" % (kmodel_src, os.path.getsize(kmodel_src) / 1048576.0), "convert", 100)

    # ---- 6 输出
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = cfg.get("tag") or cfg["model"]
    folder = os.path.join(out, "%s_%d_%s" % (tag, int(cfg["imgsz"]), stamp))
    os.makedirs(folder, exist_ok=True)
    dst_kmodel = os.path.join(folder, "best.kmodel")
    shutil.copy(kmodel_src, dst_kmodel)

    # 板子上的目录名必须是纯英文：K230 的 FAT 文件系统对中文路径支持不可靠，
    # 电脑上的输出目录可以带中文，但抄进 k230_main.py 的路径不能带。
    safe_tag = "".join(c for c in tag if c.isascii() and (c.isalnum() or c in "_-"))
    if not safe_tag:
        safe_tag = cfg["model"]
    board_folder = "%s_%d_%s" % (safe_tag, int(cfg["imgsz"]), stamp)

    snippet = '''    {
        "name": "%s",
        "family": "%s",
        "model_path": "/sdcard/kmodel/%s/best.kmodel",
        "labels": %s,
        "input_size": [%d, %d],
        "conf": 0.30,
        "nms": 0.45,
        "max_boxes": 20,
    },
''' % (
        tag,
        "yolo11" if cfg["model"].startswith("yolo11") else "yolov8",
        board_folder,
        json.dumps(list(names), ensure_ascii=False),
        int(cfg["imgsz"]), int(cfg["imgsz"]),
    )
    readme = os.path.join(folder, "使用说明.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("""K230 模型工厂 输出
=====================================
模型      : %s
输入尺寸  : %dx%d
类别      : %s
训练样本  : train %d / valid %d / test %d
生成时间  : %s

【部署步骤】
1) 把 best.kmodel 拷到 K230 板子（板子插上后会出现的 CanMV 虚拟U盘）：
     放到  /sdcard/kmodel/%s/best.kmodel

2) 打开 k230_main.py，在开头的 YOLO_MODEL_PROFILES 里加入下面这段：

%s
3) 上板运行，进入「YOLO识别」页，点「下个模型」切到「%s」，
   等状态变成「已加载」，调 CONF/NMS，最后点「保存参数」。

【注意】
- input_size 必须和转换时一致（这里是 %d）
- labels 顺序必须和训练时一致。填错的后果：模型照样能跑，
  但输出的类别名全是错的（比如对着猫喊"狐狸"）
- 板子上的路径【不要用中文】。K230 的 FAT 文件系统对中文文件名
  支持不可靠，可能压根打不开。上面给的路径已经是纯英文的
- 官方示例用的是 /sdcard/...，如果你的板子把 SD 卡挂在别处
  （比如 /data/...），把前缀改成你板子上实际的挂载点
- 板子固件的 nncase 版本要和转换工具匹配（本工具 nncase 2.11，对应 CanMV 固件 v1.8）
""" % (
            cfg["model"], int(cfg["imgsz"]), int(cfg["imgsz"]),
            "、".join(names),
            counts.get("train", 0), counts.get("valid", 0), counts.get("test", 0),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            board_folder,
            snippet,
            tag,
            int(cfg["imgsz"]),
        ))

    log("=" * 46, "done")
    log("全部完成！用时 %.1f 分钟" % ((time.time() - t0) / 60.0), "done", 100)
    log("输出目录：%s" % folder, "done")
    log("  best.kmodel   %.2f MB" % (os.path.getsize(dst_kmodel) / 1048576.0), "done")
    log("  使用说明.txt  （含可直接粘贴的配置片段）", "done")
    return folder


# ================================================================== GUI
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("K230 模型工厂 — 图片数据集 → K230 可用的 kmodel")
    root.geometry("880x660")

    pad = {"padx": 8, "pady": 4}
    frm = ttk.Frame(root)
    frm.pack(fill="both", expand=True, padx=10, pady=8)

    row = 0

    def add_row(label, widget):
        nonlocal row
        ttk.Label(frm, text=label, width=14, anchor="e").grid(row=row, column=0, sticky="e", **pad)
        widget.grid(row=row, column=1, columnspan=3, sticky="we", **pad)
        row += 1
        return widget

    # 数据集
    v_ds = tk.StringVar(value=os.path.join(WS, "datasets", "水果示例", "data"))
    e_ds = ttk.Entry(frm, textvariable=v_ds)
    add_row("数据集目录", e_ds)
    ttk.Button(frm, text="浏览…", command=lambda: v_ds.set(filedialog.askdirectory() or v_ds.get())) \
        .grid(row=row - 1, column=4, **pad)

    # 类别名
    v_names = tk.StringVar(value="apple,banana,pear")
    add_row("类别名", ttk.Entry(frm, textvariable=v_names))
    ttk.Label(frm, text="（逗号分隔，顺序必须和标注时一致）", foreground="#666") \
        .grid(row=row - 1, column=4, sticky="w", **pad)

    # 模型
    v_model = tk.StringVar(value="yolo11n")
    add_row("模型", ttk.Combobox(frm, textvariable=v_model, values=MODEL_CHOICES, state="readonly"))

    # 分辨率
    v_imgsz = tk.StringVar(value="320")
    add_row("输入分辨率", ttk.Combobox(frm, textvariable=v_imgsz,
                                   values=[str(x) for x in IMGSZ_CHOICES], state="readonly"))

    # 轮数 / batch
    sub = ttk.Frame(frm)
    sub.grid(row=row, column=1, columnspan=3, sticky="we", **pad)
    ttk.Label(frm, text="训练轮数", width=14, anchor="e").grid(row=row, column=0, sticky="e", **pad)
    v_epochs = tk.StringVar(value="100")
    ttk.Entry(sub, textvariable=v_epochs, width=8).pack(side="left")
    ttk.Label(sub, text="   batch").pack(side="left")
    v_batch = tk.StringVar(value="32")
    ttk.Entry(sub, textvariable=v_batch, width=8).pack(side="left")
    ttk.Label(sub, text="   （显存不够就调小 batch）", foreground="#666").pack(side="left")
    row += 1

    # 权重目录
    v_w = tk.StringVar(value=os.path.join(WS, "weights"))
    add_row("预训练权重目录", ttk.Entry(frm, textvariable=v_w))

    # 输出目录
    v_out = tk.StringVar(value=os.path.join(WS, "outputs"))
    add_row("输出目录", ttk.Entry(frm, textvariable=v_out))
    ttk.Button(frm, text="浏览…", command=lambda: v_out.set(filedialog.askdirectory() or v_out.get())) \
        .grid(row=row - 1, column=4, **pad)

    # 工作目录
    v_work = tk.StringVar(value=os.path.join(WS, "work"))
    add_row("工作目录", ttk.Entry(frm, textvariable=v_work))
    ttk.Label(frm, text="（中间产物：切分后的数据集、训练结果）", foreground="#666") \
        .grid(row=row - 1, column=4, sticky="w", **pad)

    frm.columnconfigure(1, weight=1)

    # 按钮
    btns = ttk.Frame(root)
    btns.pack(fill="x", padx=10)
    b_start = ttk.Button(btns, text="▶  一键生成 kmodel")
    b_start.pack(side="left", padx=6, pady=6)
    b_open = ttk.Button(btns, text="打开输出目录",
                        command=lambda: os.startfile(v_out.get()) if os.path.isdir(v_out.get()) else None)
    b_open.pack(side="left", padx=6)
    v_status = tk.StringVar(value="就绪")
    ttk.Label(btns, textvariable=v_status).pack(side="left", padx=14)
    prog = ttk.Progressbar(btns, mode="determinate", length=220, maximum=100)
    prog.pack(side="right", padx=8)

    # 日志
    logbox = scrolledtext.ScrolledText(root, height=22, font=("Consolas", 9))
    logbox.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    state = {"proc": None}

    def append(text):
        logbox.insert("end", text + "\n")
        logbox.see("end")

    def worker(cfg):
        try:
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            p = subprocess.Popen([PY_EXE, os.path.abspath(__file__), "--run", cfg],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace",
                                 env=env, creationflags=_NO_WINDOW)
            state["proc"] = p
            for line in p.stdout:
                line = line.rstrip()
                if line.startswith("@@"):
                    try:
                        rec = json.loads(line[2:])
                    except Exception:
                        append(line)
                        continue
                    append("[%s] %s" % (rec.get("t", ""), rec.get("msg", "")))
                    if rec.get("stage") == "train" and rec.get("pct") is not None:
                        prog["value"] = rec["pct"]
                else:
                    append(line)
            p.wait()
            state["proc"] = None
            if p.returncode == 0:
                root.after(0, lambda: v_status.set("✅ 完成"))
                root.after(0, lambda: messagebox.showinfo("完成", "kmodel 已生成！\n\n输出目录：\n" + v_out.get()))
            else:
                root.after(0, lambda: v_status.set("❌ 失败（看日志）"))
        except Exception:
            append(traceback.format_exc())
            root.after(0, lambda: v_status.set("❌ 异常"))

    def on_start():
        ds = v_ds.get().strip()
        if not os.path.isdir(ds):
            messagebox.showerror("错误", "数据集目录不存在：\n" + ds)
            return
        names = [x.strip() for x in v_names.get().replace("，", ",").split(",") if x.strip()]
        if not names:
            messagebox.showerror("错误", "请填写类别名")
            return
        cfg = {
            "dataset_dir": ds,
            "names": names,
            "model": v_model.get(),
            "imgsz": int(v_imgsz.get()),
            "epochs": int(v_epochs.get()),
            "batch": int(v_batch.get()),
            "weights_root": v_w.get(),
            "work_dir": v_work.get(),
            "out_dir": v_out.get(),
            "tag": names[0] if names else "model",
        }
        cfg_path = os.path.join(HERE, "last_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        logbox.delete("1.0", "end")
        append("配置：")
        append(json.dumps(cfg, ensure_ascii=False, indent=2))
        append("-" * 60)
        v_status.set("运行中…")
        prog["value"] = 0
        import threading
        threading.Thread(target=worker, args=(cfg_path,), daemon=True).start()

    b_start.config(command=on_start)

    append("K230 模型工厂  —  图片数据集 → K230 可用的 kmodel")
    append("流程：数据校验 → 切分 → 训练 → 导出ONNX → 转kmodel → 输出配置片段")
    append("")
    append("数据集需要是 YOLO 格式：<目录>/images/*.jpg + <目录>/labels/*.txt")
    append("标签 txt 每行： 类别号  中心x  中心y  宽  高   （都已归一化到 0~1）")
    append("")
    append("填好上面几项，点「一键生成 kmodel」即可。")
    append("")

    root.mainloop()


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="无界面流水线模式，传入 config.json 路径")
    args = ap.parse_args()

    if args.run:
        with open(args.run, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        try:
            run_pipeline(cfg)
            sys.exit(0)
        except Exception as e:
            log("错误：" + str(e), "error")
            log(traceback.format_exc(), "error")
            sys.exit(1)
    else:
        launch_gui()


if __name__ == "__main__":
    main()
