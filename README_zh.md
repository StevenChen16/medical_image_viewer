# 医学影像查看器

一个功能强大且用户友好的医学影像查看器，使用 PySide6 构建，支持 NIfTI (.nii, .nii.gz) 和 MetaImage (.mha, .mhd) 格式。它为临床研究和数据分析提供了一套全面的工具。

![logo](media/logo.png)

## 功能特性

-   **多格式支持**: 原生支持 NIfTI (.nii, .nii.gz) 和 MetaImage (.mha, .mhd) 文件。
-   **三平面视图**: 同时显示轴状、矢状和冠状视图，用于全面的三维数据分析。
-   **标签叠加**: 在基础图像上加载并叠加分割掩码。
-   **可自定义的标签颜色**: 轻松为不同的标签分配和更改颜色，以增强可视化效果。
-   **交互式导航**:
    -   使用鼠标滚轮或滑块滚动切片。
    -   使用 `Ctrl + 鼠标滚轮` 缩放。
    -   通过右键单击并拖动来平移视图。
-   **可调节的叠加透明度**: 微调标签叠加的 alpha 值以获得最佳清晰度。
-   **体数据保存**: 可将图像、标签或叠加结果导出为 NIfTI 或 MetaImage 格式，使用 `Ctrl+Shift+S` 快速打开保存菜单。
-   **屏幕截图**: 将当前视图保存为 PNG 图像。
-   **双语界面**: “关于”对话框支持中英文。
-   **命令行界面**: 从命令行直接加载图像和标签，以加快工作流程。

## 安装

1.  克隆仓库：
    ```bash
    git clone https://github.com/StevenChen16/medical_image_viewer.git
    cd medical_image_viewer
    ```

2.  安装所需的依赖项：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

从命令行运行查看器：

```bash
python nifti_viewer.py
```

您还可以指定在启动时打开的图像和标签文件：

```bash
python nifti_viewer.py -i /path/to/your/image.nii.gz -l /path/to/your/labels.nii.gz
```

### 命令行参数

-   `-i`, `--image`: 医学图像文件的路径。
-   `-l`, `--labels`: 标签掩码文件的路径。
-   `-o`, `--output`: 保存屏幕截图的默认路径。

更多选项，请运行：
```bash
python nifti_viewer.py --help
```

## 依赖

-   PySide6
-   NumPy
-   Nibabel
-   SimpleITK
-   Pillow
-   Matplotlib

具体版本请参见 `requirements.txt`。

## 开发路线图

有关未来开发计划和即将推出功能的详细信息，请参见：
- [ROADMAP_zh.md](ROADMAP_zh.md) - 综合开发路线图
- [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md) - 技术实施细节（英文）

### 下一步主要功能
- 增强测量工具（面积、体积、角度）
- DICOM格式支持
- 窗宽窗位调节与预设
- 可扩展的插件架构
- 高级3D可视化功能
