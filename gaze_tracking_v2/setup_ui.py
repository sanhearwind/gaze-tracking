"""Small desktop setup window for the V2 camera application."""

from __future__ import annotations

import argparse
from typing import Sequence


def configure_from_ui(
    args: argparse.Namespace,
    cameras: Sequence[tuple[int, int, int]],
) -> argparse.Namespace:
    """Show a compact Tk setup window and return the selected arguments.

    Tkinter is imported lazily so offline metrics and dependency-light tests do
    not require a graphical environment.
    """

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError(
            "配置窗口需要 Tkinter。请使用 --no-console 并明确指定参数，"
            "或安装包含 Tkinter 的 Python 发行版。"
        ) from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "配置窗口需要桌面显示环境。无界面环境请使用 --no-console "
            "并明确指定摄像头参数。"
        ) from exc
    root.title("注视追踪 V2.6 配置")
    root.resizable(False, False)

    content = ttk.Frame(root, padding=14)
    content.grid(row=0, column=0, sticky="nsew")

    camera_values = [
        f"{index}  ({width}x{height})" for index, width, height in cameras
    ]
    if not camera_values:
        camera_values = ["0  （未检测到，可尝试使用）"]

    camera_default = args.camera if args.camera is not None else (
        cameras[0][0] if cameras else 0
    )
    camera_value = tk.StringVar()
    selected_camera_value = next(
        (
            value
            for value in camera_values
            if value.split()[0] == str(camera_default)
        ),
        camera_values[0],
    )
    camera_value.set(selected_camera_value)

    selected_camera = next(
        (item for item in cameras if item[0] == camera_default),
        cameras[0] if cameras else (0, 640, 480),
    )
    width_value = tk.StringVar(value=str(args.width or selected_camera[1] or 640))
    height_value = tk.StringVar(value=str(args.height or selected_camera[2] or 480))
    condition_labels = {
        "normal": "正常",
        "glasses": "戴眼镜",
        "no_glasses": "不戴眼镜",
    }
    condition_values_by_label = {
        label: value for value, label in condition_labels.items()
    }
    raw_condition = args.condition or "normal"
    condition_value = tk.StringVar(
        value=condition_labels.get(raw_condition, raw_condition)
    )
    strict_value = tk.BooleanVar(value=bool(args.strict_pose_gate))
    randomize_value = tk.BooleanVar(value=bool(args.randomize_evaluation))
    samples_value = tk.StringVar(value=str(args.min_calibration_samples))
    alpha_value = tk.StringVar(value=str(args.filter_alpha))
    error_value = tk.StringVar()

    ttk.Label(
        content,
        text="启动前请配置摄像头和评估参数。",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    def add_row(row: int, label: str, widget: object) -> None:
        ttk.Label(content, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=4
        )
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    camera_box = ttk.Combobox(
        content, textvariable=camera_value, values=camera_values, state="readonly", width=28
    )
    add_row(1, "摄像头", camera_box)
    add_row(2, "请求宽度", ttk.Entry(content, textvariable=width_value, width=12))
    add_row(3, "请求高度", ttk.Entry(content, textvariable=height_value, width=12))

    condition_values = list(condition_values_by_label)
    if condition_value.get() not in condition_values:
        condition_values.insert(0, condition_value.get())
    condition_box = ttk.Combobox(
        content, textvariable=condition_value, values=condition_values, width=25
    )
    add_row(4, "测试条件", condition_box)
    add_row(5, "每点最少校准帧数", ttk.Entry(content, textvariable=samples_value, width=12))
    add_row(6, "EMA 滤波系数", ttk.Entry(content, textvariable=alpha_value, width=12))

    def update_size_from_camera(_event: object = None) -> None:
        try:
            index = int(camera_value.get().split()[0])
        except (TypeError, ValueError):
            return
        for camera_index, camera_width, camera_height in cameras:
            if camera_index == index:
                if not args.width:
                    width_value.set(str(camera_width or 640))
                if not args.height:
                    height_value.set(str(camera_height or 480))
                return

    camera_box.bind("<<ComboboxSelected>>", update_size_from_camera)

    strict_check = ttk.Checkbutton(
        content,
        text="启用严格头部姿态门控（相对于中性姿态）",
        variable=strict_value,
    )
    strict_check.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 2))
    ttk.Label(
        content,
        text="记录中性参考姿态时，请保持头部正对摄像头。",
        foreground="#666666",
    ).grid(row=8, column=0, columnspan=2, sticky="w")
    randomize_check = ttk.Checkbutton(
        content,
        text="随机排列评估点顺序",
        variable=randomize_value,
    )
    randomize_check.grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 2))

    ttk.Label(content, textvariable=error_value, foreground="#b00020").grid(
        row=10, column=0, columnspan=2, sticky="w", pady=(8, 2)
    )

    buttons = ttk.Frame(content)
    buttons.grid(row=11, column=0, columnspan=2, sticky="e", pady=(10, 0))

    result = {"started": False}

    def start() -> None:
        try:
            camera_index = int(camera_value.get().split()[0])
            width = int(width_value.get())
            height = int(height_value.get())
            min_samples = int(samples_value.get())
            alpha = float(alpha_value.get())
            if width <= 0 or height <= 0:
                raise ValueError("宽度和高度必须为正数")
            if min_samples < 3:
                raise ValueError("每点最少校准帧数不能小于 3")
            if not 0.0 < alpha <= 1.0:
                raise ValueError("EMA 滤波系数必须在 (0, 1] 范围内")
        except ValueError as exc:
            error_value.set(str(exc))
            return

        args.camera = camera_index
        args.width = width
        args.height = height
        selected_condition = condition_value.get().strip()
        args.condition = condition_values_by_label.get(
            selected_condition, selected_condition or "normal"
        )
        args.min_calibration_samples = min_samples
        args.filter_alpha = alpha
        args.strict_pose_gate = bool(strict_value.get())
        args.randomize_evaluation = bool(randomize_value.get())
        result["started"] = True
        root.destroy()

    def cancel() -> None:
        root.destroy()

    ttk.Button(buttons, text="启动", command=start).grid(row=0, column=0, padx=4)
    ttk.Button(buttons, text="取消", command=cancel).grid(row=0, column=1, padx=4)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.bind("<Return>", lambda _event: start())
    root.bind("<Escape>", lambda _event: cancel())
    root.mainloop()

    if not result["started"]:
        raise SystemExit(0)
    return args
