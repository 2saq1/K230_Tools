# -*- coding: utf-8 -*-
"""K230 一站式工具 —— 便携版启动器

把「标注」和「训练/转换」整合进一个绿色文件夹，拷到 U 盘即可用。

关键设计：
1. 所有路径由本文件位置动态推导，换电脑/换盘符都不受影响
2. 若包路径含中文，自动创建 ASCII 目录联接（C:\\K230Tools_xxxxxxxx）
   并让所有子程序走联接路径 —— nncase 的 .NET 加载器不认中文路径
3. 启动标注前自动清空 labelImg 记住的上次目录
"""
import os
import sys
import json
import time
import pickle
import hashlib
import subprocess

REAL_ROOT = os.path.dirname(os.path.abspath(__file__))

_NO_WINDOW = 0x08000000


def ascii_root(root):
    """包路径含非 ASCII 字符时，返回一个指向它的 ASCII 目录联接路径。"""
    try:
        root.encode("ascii")
        return root
    except UnicodeEncodeError:
        drive = os.environ.get("SystemDrive", "C:") + "\\"
        junction = drive + "K230Tools_" + hashlib.md5(root.encode("utf-8")).hexdigest()[:8]
        if not os.path.isdir(junction):
            try:
                subprocess.run(["cmd", "/c", "mklink", "/J", junction, root],
                               capture_output=True, timeout=30)
            except Exception:
                pass
        return junction if os.path.isdir(junction) else root


ROOT = ascii_root(REAL_ROOT)
PYDIR = os.path.join(ROOT, "python")
PY = os.path.join(PYDIR, "python.exe")
PYW = os.path.join(PYDIR, "pythonw.exe")
WS = os.path.join(ROOT, "workspace")
DATASETS = os.path.join(WS, "datasets")
FACTORY = os.path.join(ROOT, "apps", "factory", "k230_model_factory.py")
FACTORY_CFG = os.path.join(ROOT, "apps", "factory", "last_config.json")
README = os.path.join(ROOT, "使用说明.txt")
LOG = os.path.join(WS, "launcher.log")
SETTINGS = os.path.join(os.path.expanduser("~"), ".labelImgSettings.pkl")


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except Exception:
        pass


def child_env():
    env = dict(os.environ)
    env["PATH"] = PYDIR + ";" + os.path.join(PYDIR, "DLLs") + ";" + env.get("PATH", "")
    # nncase 需要 .NET 7，包内自带，强制用它（ROOT 是 ASCII 联接路径）
    d = os.path.join(ROOT, "dotnet")
    if os.path.isdir(d):
        env["DOTNET_ROOT"] = d
    return env


def clear_labelimg_history():
    """清掉 labelImg 记住的上次目录（窗口大小保留）。"""
    try:
        if not os.path.isfile(SETTINGS):
            return
        import PyQt5.QtCore  # noqa: F401
        data = pickle.load(open(SETTINGS, "rb"))
        if isinstance(data, dict):
            for k in ("savedir", "lastOpenDir", "recentFiles", "filename"):
                data.pop(k, None)
            pickle.dump(data, open(SETTINGS, "wb"), pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def ensure_dataset(ds, names):
    """补齐 images/labels/classes.txt 三件套。"""
    os.makedirs(os.path.join(ds, "images"), exist_ok=True)
    os.makedirs(os.path.join(ds, "labels"), exist_ok=True)
    cf = os.path.join(ds, "classes.txt")
    if not os.path.isfile(cf) or os.path.getsize(cf) == 0:
        open(cf, "w", encoding="utf-8").write("\n".join(names) + "\n")


def list_datasets():
    if not os.path.isdir(DATASETS):
        return []
    out = []
    for d in sorted(os.listdir(DATASETS)):
        p = os.path.join(DATASETS, d)
        if os.path.isdir(p):
            if os.path.isdir(os.path.join(p, "images")):
                out.append(p)
            else:
                for sub in sorted(os.listdir(p)):
                    q = os.path.join(p, sub)
                    if os.path.isdir(q) and os.path.isdir(os.path.join(q, "images")):
                        out.append(q)
    return out


# ================================================================== GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("K230 一站式工具 —— 标注 + 训练（便携版）")
        self.root.geometry("720x360")
        self.root.resizable(True, True)

        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=14, pady=10)

        ttk.Label(frm, text="K230 一站式工具", font=("Microsoft YaHei UI", 16, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(frm, text="绿色便携版 · 标注 + 训练 + 转 kmodel，插上就能用",
                  foreground="#666666").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

        # ---- 数据集
        ttk.Label(frm, text="数据集目录").grid(row=2, column=0, sticky="w", **pad)
        self.v_ds = tk.StringVar()
        self.cb_ds = ttk.Combobox(frm, textvariable=self.v_ds, width=62)
        self.cb_ds.grid(row=2, column=1, columnspan=2, sticky="we", **pad)
        ds_list = list_datasets()
        if ds_list:
            self.cb_ds["values"] = ds_list
            self.v_ds.set(ds_list[0])
        else:
            self.v_ds.set(os.path.join(DATASETS, "我的数据集"))
        self.btn_new = ttk.Button(frm, text="＋ 新建数据集", command=self.new_dataset)
        self.btn_new.grid(row=3, column=1, sticky="w", **pad)
        self.btn_browse = ttk.Button(frm, text="浏览…", command=self.browse)
        self.btn_browse.grid(row=3, column=2, sticky="w", **pad)

        # ---- 类别
        ttk.Label(frm, text="类别名（逗号分隔，顺序=编号）").grid(row=4, column=0, sticky="w", **pad)
        self.v_names = tk.StringVar(value="apple,banana,pear")
        ttk.Entry(frm, textvariable=self.v_names, width=62).grid(row=4, column=1, columnspan=2, sticky="we", **pad)

        # ---- 按钮
        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=3, sticky="we", pady=18)
        self.b_lab = ttk.Button(btns, text="🏷  打开标注工具", command=self.launch_label, width=20)
        self.b_lab.pack(side="left", padx=6)
        self.b_fac = ttk.Button(btns, text="🚀  打开模型工厂（训练/转换）", command=self.launch_factory, width=26)
        self.b_fac.pack(side="left", padx=6)
        self.b_doc = ttk.Button(btns, text="📖  使用说明", command=self.show_readme, width=14)
        self.b_doc.pack(side="left", padx=6)

        # ---- 状态
        self.txt = tk.Text(frm, height=6, wrap="word", state="disabled",
                           font=("Consolas", 9))
        self.txt.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(6, weight=1)
        frm.columnconfigure(1, weight=1)

        self.say("就绪。先选数据集和类别名，再点上面的按钮。")
        if os.path.isdir(junction := os.path.join(os.environ.get("SystemDrive", "C:") + "\\", "K230Tools_" + hashlib.md5(REAL_ROOT.encode("utf-8")).hexdigest()[:8])):
            self.say("检测到包路径含中文，已通过 ASCII 联接运行：%s" % junction)

    def say(self, msg):
        self.txt.configure(state="normal")
        self.txt.insert("end", msg.rstrip() + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def get_ds(self):
        ds = self.v_ds.get().strip().strip('"')
        if not ds:
            messagebox.showwarning("提示", "先填数据集目录")
            return None
        return ds

    def get_names(self):
        names = [n.strip() for n in self.v_names.get().split(",") if n.strip()]
        if not names:
            messagebox.showwarning("提示", "先填类别名，比如 apple,banana,pear")
            return None
        return names

    def new_dataset(self):
        name = simpledialog.askstring("新建数据集", "数据集名字（会建在 workspace\\datasets 下）：",
                                      parent=self.root)
        if not name:
            return
        ds = os.path.join(DATASETS, name)
        names = self.get_names() or []
        ensure_dataset(ds, names)
        self.v_ds.set(ds)
        self.cb_ds["values"] = list_datasets()
        self.say("已创建数据集：%s" % ds)

    def browse(self):
        d = filedialog.askdirectory(parent=self.root, title="选数据集目录（含 images/labels 的那一层）")
        if d:
            self.v_ds.set(d)

    def launch_label(self):
        ds = self.get_ds()
        names = self.get_names()
        if not ds or names is None:
            return
        ensure_dataset(ds, names)
        img = os.path.join(ds, "images")
        sav = os.path.join(ds, "labels")
        cls = os.path.join(ds, "classes.txt")
        clear_labelimg_history()
        log("启动 labelImg: %s" % ds)
        try:
            subprocess.Popen([PY, "-m", "labelImg.labelImg", img, cls, sav],
                             cwd=ROOT, env=child_env(), creationflags=_NO_WINDOW)
            self.say("已打开标注工具\n  图片: %s\n  标签: %s" % (img, sav))
        except Exception as e:
            messagebox.showerror("失败", "标注工具启动失败：%s\n看 workspace\\launcher.log" % e)
            log("labelImg 失败: %s" % e)

    def launch_factory(self):
        ds = self.get_ds()
        names = self.get_names()
        if not ds or names is None:
            return
        ensure_dataset(ds, names)
        cfg = {
            "dataset_dir": ds.replace("\\", "/"),
            "names": names,
            "model": "yolo11n",
            "imgsz": 320,
            "epochs": 100,
            "batch": 32,
            "weights_root": os.path.join(WS, "weights"),
            "work_dir": os.path.join(WS, "work"),
            "out_dir": os.path.join(WS, "outputs"),
            "tag": os.path.basename(os.path.normpath(ds)),
        }
        try:
            json.dump(cfg, open(FACTORY_CFG, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("失败", "写配置失败：%s" % e)
            return
        log("启动模型工厂: %s (%d 类)" % (ds, len(names)))
        try:
            subprocess.Popen([PYW, FACTORY], cwd=os.path.dirname(FACTORY),
                             env=child_env())
            self.say("已打开模型工厂\n  数据集: %s\n  类别: %s" % (ds, ", ".join(names)))
        except Exception as e:
            messagebox.showerror("失败", "模型工厂启动失败：%s" % e)
            log("工厂失败: %s" % e)

    def show_readme(self):
        if os.path.isfile(README):
            try:
                os.startfile(README)
                return
            except Exception:
                pass
        try:
            with open(README, encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            txt = "找不到 使用说明.txt"
        top = tk.Toplevel(self.root)
        top.title("使用说明")
        t = tk.Text(top, wrap="word", width=90, height=32)
        t.pack(fill="both", expand=True, padx=8, pady=8)
        t.insert("1.0", txt)
        t.configure(state="disabled")


def main():
    log("-" * 46)
    try:
        log("启动器启动 REAL_ROOT=%s ROOT=%s" % (REAL_ROOT, ROOT))
        app = App()
        app.root.mainloop()
    except Exception:
        # 在陌生的电脑上万一出错，也要让用户看得到，而不是双击无反应
        import traceback
        msg = traceback.format_exc()
        try:
            log("致命错误:\n" + msg)
        except Exception:
            pass
        try:
            import tkinter as tk
            from tkinter import messagebox
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror("K230 工具箱 启动失败", msg[-900:])
            r.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
