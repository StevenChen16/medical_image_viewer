#!/usr/bin/env python3
"""
NIfTI/MHA Viewer with PySide 6
A medical image viewer for NIfTI and MHA format files.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any
import traceback

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

# Conditionally import SimpleITK for MHA support
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSlider, QLabel, QCheckBox, QPushButton, QMenuBar, QStatusBar,
    QDockWidget, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QToolTip, QFrame, QLineEdit,
    QDialog, QTabWidget, QTextEdit, QScrollArea, QSizePolicy
)
from PySide6.QtCore import (
    Qt, QObject, Signal, QThread, QRunnable, QThreadPool, QTimer,
    QRect, QSize, QPointF
)
from PySide6.QtGui import (
    QPixmap, QPainter, QImage, QPen, QBrush, QColor, QTransform,
    QAction, QIcon, QFont, QWheelEvent, QMouseEvent, QPixmapCache
)


# ============================================================================
# MODEL LAYER - Data Management & Caching
# ============================================================================

class ImageModel(QObject):
    """
    Core data model for medical image and label management.
    Handles loading, caching, and vectorized rendering for NIfTI and MHA.
    """
    
    # Signals
    imageLoaded = Signal(str, tuple)  # filename, shape
    labelLoaded = Signal(str, int)    # filename, unique_labels_count
    sliceReady = Signal(str, int, QImage)  # view_name, slice_idx, image
    loadError = Signal(str)           # error_message
    loadProgress = Signal(int)        # loading progress percentage
    
    def __init__(self):
        super().__init__()
        
        # Core data storage
        self.image_data: Optional[np.ndarray] = None
        self.label_data: Optional[np.ndarray] = None
        
        # File paths
        self.image_path: str = ""
        self.label_path: str = ""
        
        # View configurations
        self.view_configs = {
            'axial': {'axis': 2, 'slice': 0, 'show': True, 'overlay': True, 'rotation': 0, 'alpha': 0.5},
            'sagittal': {'axis': 0, 'slice': 0, 'show': True, 'overlay': True, 'rotation': 0, 'alpha': 0.5},
            'coronal': {'axis': 1, 'slice': 0, 'show': True, 'overlay': True, 'rotation': 0, 'alpha': 0.5}
        }
        
        # Global settings
        self.global_overlay = True
        self.global_alpha = 0.5

        # Pre-compute label colormap (labels 1-20)
        cmap = plt.get_cmap('tab20')(np.linspace(0, 1, 20))[:, :3]
        self._label_cmap = (cmap * 255).astype(np.uint8)
        
    def _load_image_data(self, filepath: str) -> np.ndarray:
        """
        Internal helper to load medical image data from various formats.
        Returns a NumPy array in (X, Y, Z) orientation.
        """
        file_ext = "".join(Path(filepath).suffixes).lower()

        if file_ext in ['.nii', '.nii.gz']:
            img = nib.load(filepath)
            return img.get_fdata(dtype=np.float32)
        
        elif file_ext in ['.mha', '.mhd']:
            if not SITK_AVAILABLE:
                raise ImportError("SimpleITK is required to load MHA/MHD files. Please run: pip install SimpleITK")
            
            itk_img = sitk.ReadImage(filepath)
            # Transpose from ITK's (Z, Y, X) to nibabel's (X, Y, Z) convention
            np_array = sitk.GetArrayFromImage(itk_img)
            return np_array.transpose(2, 1, 0).astype(np.float32)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @lru_cache(maxsize=50)
    def get_slice_data(self, axis: int, slice_idx: int, is_label: bool = False) -> np.ndarray:
        """Extract slice from 3D data with caching."""
        data = self.label_data if is_label else self.image_data
        if data is None:
            return np.array([])
            
        slice_idx = max(0, min(slice_idx, data.shape[axis] - 1))
        
        if axis == 0:  # Sagittal
            return data[slice_idx, :, :]
        elif axis == 1:  # Coronal
            return data[:, slice_idx, :]
        else:  # Axial
            return data[:, :, slice_idx]
    
    def normalize_image(self, img_slice: np.ndarray) -> np.ndarray:
        """Vectorized image normalization to 0-255 range."""
        if img_slice.size == 0:
            return img_slice
            
        img_min, img_max = np.min(img_slice), np.max(img_slice)
        if img_max > img_min:
            normalized = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img_slice, dtype=np.uint8)
        return normalized
    
    def create_label_overlay(self, label_slice: np.ndarray) -> Optional[np.ndarray]:
        """Create vectorized colored overlay from label slice."""
        if label_slice.size == 0:
            return None

        labels = label_slice.astype(np.intp, copy=False)
        mask = labels > 0
        if not np.any(mask):
            return None

        overlay = np.zeros((*label_slice.shape, 3), dtype=np.uint8)
        overlay[mask] = self._label_cmap[(labels[mask] - 1) % len(self._label_cmap)]
        return overlay

    def render_slice(self, view_name: str, slice_idx: int,
                    show_overlay: bool = True, alpha: float = 0.5) -> QImage:
        """Public rendering interface with LRU caching."""
        rotation = self.view_configs[view_name]['rotation']
        return self._render_slice_cached(view_name, slice_idx,
                                         show_overlay and self.global_overlay,
                                         alpha, rotation)

    @lru_cache(maxsize=100)
    def _render_slice_cached(self, view_name: str, slice_idx: int,
                              show_overlay: bool, alpha: float,
                              rotation: int) -> QImage:
        """Internal slice renderer that is LRU cached."""
        config = self.view_configs[view_name]
        axis = config['axis']

        img_slice = self.get_slice_data(axis, slice_idx, False)
        if img_slice.size == 0:
            return QImage()

        img_normalized = self.normalize_image(img_slice)

        # Handle overlay if available
        if show_overlay and self.label_data is not None:
            label_slice = self.get_slice_data(axis, slice_idx, True)
            overlay = self.create_label_overlay(label_slice)
            if overlay is not None:
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
                mask = label_slice > 0
                img_rgb = img_rgb.astype(np.float32, copy=False)
                overlay_f = overlay.astype(np.float32, copy=False)
                img_rgb[mask] = (1 - alpha) * img_rgb[mask] + alpha * overlay_f[mask]
                img_rgb = img_rgb.astype(np.uint8, copy=False)
                img_rgb = np.flipud(img_rgb)
                img_rgb = np.ascontiguousarray(img_rgb)
                height, width, _ = img_rgb.shape
                bytes_per_line = 3 * width
                qimage = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                img_flipped = np.flipud(img_normalized)
                img_flipped = np.ascontiguousarray(img_flipped)
                height, width = img_flipped.shape
                qimage = QImage(img_flipped.data, width, height, img_flipped.strides[0], QImage.Format_Grayscale8)
        else:
            img_flipped = np.flipud(img_normalized)
            img_flipped = np.ascontiguousarray(img_flipped)
            height, width = img_flipped.shape
            qimage = QImage(img_flipped.data, width, height, img_flipped.strides[0], QImage.Format_Grayscale8)

        if rotation != 0 and not qimage.isNull():
            transform = QTransform()
            transform.rotate(rotation)
            qimage = qimage.transformed(transform)

        return qimage
    
    def load_image(self, filepath: str):
        """Load medical image data."""
        try:
            self.loadProgress.emit(0)
            self.image_data = self._load_image_data(filepath)
            self.image_path = filepath
            self.loadProgress.emit(70)

            # Clear cache when new data is loaded
            self.get_slice_data.cache_clear()
            self._render_slice_cached.cache_clear()
            QPixmapCache.clear()

            # Reset slice positions
            for config in self.view_configs.values():
                config['slice'] = self.image_data.shape[config['axis']] // 2

            self.loadProgress.emit(100)
            self.imageLoaded.emit(filepath, self.image_data.shape)

        except Exception as e:
            self.loadError.emit(f"Failed to load image: {str(e)}")

    def load_labels(self, filepath: str):
        """Load medical label data."""
        try:
            self.loadProgress.emit(0)
            new_label_data = self._load_image_data(filepath).astype(np.int32)
            self.loadProgress.emit(70)

            # Validate shape compatibility
            if (
                self.image_data is not None and
                new_label_data.shape != self.image_data.shape):
                self.loadError.emit(
                    f"Label shape {new_label_data.shape} doesn't match "
                    f"image shape {self.image_data.shape}")
                return

            self.label_data = new_label_data
            self.label_path = filepath

            # Clear cache
            self.get_slice_data.cache_clear()
            self._render_slice_cached.cache_clear()
            QPixmapCache.clear()

            unique_labels = len(np.unique(self.label_data))
            self.loadProgress.emit(100)
            self.labelLoaded.emit(filepath, unique_labels)

        except Exception as e:
            self.loadError.emit(f"Failed to load labels: {str(e)}")
    
    def get_max_slice(self, view_name: str) -> int:
        """Get maximum slice index for a view."""
        if self.image_data is None:
            return 0
        axis = self.view_configs[view_name]['axis']
        return self.image_data.shape[axis] - 1
    
    def set_slice(self, view_name: str, slice_idx: int):
        """Update slice index for a view."""
        max_slice = self.get_max_slice(view_name)
        self.view_configs[view_name]['slice'] = max(0, min(slice_idx, max_slice))
    
    def get_label_at_position(self, view_name: str, x: int, y: int) -> int:
        """Get label value at specific pixel position."""
        if self.label_data is None:
            return 0
            
        config = self.view_configs[view_name]
        slice_idx = config['slice']
        label_slice = self.get_slice_data(config['axis'], slice_idx, True)
        
        if label_slice.size == 0:
            return 0
            
        # Account for image flipping and bounds checking
        h, w = label_slice.shape
        y = h - 1 - y  # Flip Y coordinate
        
        if 0 <= x < w and 0 <= y < h:
            return int(label_slice[y, x])
        return 0


# ============================================================================
# WORKER LAYER - Background Loading
# ============================================================================

class LoadWorker(QRunnable):
    """Background worker for non-blocking file loading."""
    
    def __init__(self, model: ImageModel, filepath: str, is_label: bool = False):
        super().__init__()
        self.model = model
        self.filepath = filepath
        self.is_label = is_label
    
    def run(self):
        """Execute loading in background thread."""
        try:
            if self.is_label:
                self.model.load_labels(self.filepath)
            else:
                self.model.load_image(self.filepath)
        except Exception as e:
            self.model.loadError.emit(f"Background loading failed: {str(e)}")


# ============================================================================
# VIEW LAYER - UI Components
# ============================================================================

class AboutDialog(QDialog):
    """
    About dialog with bilingual (English/Chinese) support and software information.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Medical Image Viewer")
        self.setFixedSize(600, 500)
        self.setModal(True)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Tab widget for language switching
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_english_tab()
        self.create_chinese_tab()
        
        # Close button
        close_btn = QPushButton("Close / 关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        # Apply dialog styles
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #3c3c3c;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #4a4a4a;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            QTextEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                color: white;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
    
    def create_english_tab(self):
        """Create English content tab."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Main title
        title_label = QLabel("Medical Image Viewer")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a90e2; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Version info
        version_label = QLabel("Version 0.1.0")
        version_label.setStyleSheet("font-size: 14px; color: #ccc; margin-bottom: 15px;")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        # Description
        description = QTextEdit()
        description.setReadOnly(True)
        description.setMaximumHeight(300)
        description.setHtml("""
        <h3 style=\"color: #4a90e2;\">What is this Viewer?</h3>
        <p>A simple, fast viewer for medical images. View MRI scans, overlays, and segmentation masks 
        in three perspectives simultaneously.</p>
        
        <h3 style=\"color: #4a90e2;\">How to Use</h3>
        <ul>
            <li><b>Load Files:</b> Use File menu → Load Image/Labels, or type paths in the right panel</li>
            <li><b>Navigate:</b> Mouse wheel scrolls through slices, Ctrl+wheel zooms in/out</li>
            <li><b>Pan & Rotate:</b> Right-click drag to move view, click rotation buttons to flip</li>
            <li><b>Overlays:</b> Check "Show Overlay" and adjust transparency slider</li>
            <li><b>Save:</b> File → Save Screenshot or Ctrl+S</li>
        </ul>
        
        <h3 style=\"color: #4a90e2;\">Keyboard Shortcuts</h3>
        <table style=\"width: 100%; border-collapse: collapse;">
            <tr><td style=\"padding: 4px;\"><b>Ctrl+O</b></td><td style=\"padding: 4px;\">Open image file</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+L</b></td><td style=\"padding: 4px;\">Open label file</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+S</b></td><td style=\"padding: 4px;\">Save screenshot</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+R</b></td><td style=\"padding: 4px;\">Reset views and clear cache</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+T</b></td><td style=\"padding: 4px;\">Toggle control panel</td></tr>
            <tr><td style=\"padding: 4px;\"><b>F</b></td><td style=\"padding: 4px;\">Fit all views to window</td></tr>
            <tr><td style=\"padding: 4px;\"><b>F1</b></td><td style=\"padding: 4px;\">Show this dialog</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Wheel</b></td><td style=\"padding: 4px;\">Scroll through slices</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+Wheel</b></td><td style=\"padding: 4px;\">Zoom in/out</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Right-drag</b></td><td style=\"padding: 4px;\">Pan view</td></tr>
        </table>
        
        <h3 style=\"color: #4a90e2;\">Command Line Options</h3>
        <p>Run <code>python nifti_viewer.py --help</code> for full options.<br>
        Examples: <code>-i image.mha -l labels.nii.gz</code></p>
        
        <h3 style=\"color: #4a90e2;\">File Support</h3>
        <p>Supports NIfTI (.nii, .nii.gz) and MetaImage (.mha, .mhd) formats.</p>
        """)
        layout.addWidget(description)
        
        scroll_area.setWidget(content_widget)
        self.tab_widget.addTab(scroll_area, "English")
    
    def create_chinese_tab(self):
        """Create Chinese content tab."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Main title
        title_label = QLabel("医学影像查看器")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a90e2; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Version info
        version_label = QLabel("版本 0.1.0")
        version_label.setStyleSheet("font-size: 14px; color: #ccc; margin-bottom: 15px;")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        # Description
        description = QTextEdit()
        description.setReadOnly(True)
        description.setMaximumHeight(300)
        description.setHtml("""
        <h3 style=\"color: #4a90e2;\">什么是医学影像查看器？</h3>
        <p>简单快速的医学影像查看工具。支持查看磁共振(MRI)扫描图像、叠加图层和分割掩膜，
        同时显示三个角度的切面视图。</p>
        
        <h3 style=\"color: #4a90e2;\">如何使用</h3>
        <ul>
            <li><b>加载文件：</b>使用文件菜单 → 加载影像/标签，或在右侧面板输入文件路径</li>
            <li><b>导航操作：</b>鼠标滚轮切换切片，Ctrl+滚轮缩放视图</li>
            <li><b>平移旋转：</b>右键拖拽移动视图，点击旋转按钮翻转方向</li>
            <li><b>叠加显示：</b>勾选"显示叠加"并调节透明度滑条</li>
            <li><b>保存截图：</b>文件 → 保存截图或按 Ctrl+S</li>
        </ul>
        
        <h3 style=\"color: #4a90e2;\">快捷键一览</h3>
        <table style=\"width: 100%; border-collapse: collapse;">
            <tr><td style=\"padding: 4px;\"><b>Ctrl+O</b></td><td style=\"padding: 4px;\">打开影像文件</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+L</b></td><td style=\"padding: 4px;\">打开标签文件</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+S</b></td><td style=\"padding: 4px;\">保存截图</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+R</b></td><td style=\"padding: 4px;\">重置视图并清空缓存</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+T</b></td><td style=\"padding: 4px;\">切换控制面板显示</td></tr>
            <tr><td style=\"padding: 4px;\"><b>F</b></td><td style=\"padding: 4px;\">适应所有视图到窗口</td></tr>
            <tr><td style=\"padding: 4px;\"><b>F1</b></td><td style=\"padding: 4px;\">显示此对话框</td></tr>
            <tr><td style=\"padding: 4px;\"><b>滚轮</b></td><td style=\"padding: 4px;\">切换切片</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+滚轮</b></td><td style=\"padding: 4px;\">缩放视图</td></tr>
            <tr><td style=\"padding: 4px;\"><b>右键拖拽</b></td><td style=\"padding: 4px;\">平移视图</td></tr>
        </table>
        
        <h3 style=\"color: #4a90e2;\">命令行选项</h3>
        <p>运行 <code>python nifti_viewer.py --help</code> 查看完整选项。<br>
        示例：<code>-i image.mha -l labels.nii.gz</code></p>
        
        <h3 style=\"color: #4a90e2;\">文件支持</h3>
        <p>支持 NIfTI (.nii, .nii.gz) 和 MetaImage (.mha, .mhd) 格式。</p>
        """)
        layout.addWidget(description)
        
        scroll_area.setWidget(content_widget)
        self.tab_widget.addTab(scroll_area, "中文")


class SliceView(QGraphicsView):
    """
    Enhanced QGraphicsView for medical image slice display.
    Supports zooming, panning, rotation, and label tooltips.
    """
    
    # Signals
    mousePositionChanged = Signal(int, int)  # x, y coordinates
    wheelScrolled = Signal(int)              # delta for slice navigation
    
    def __init__(self, view_name: str, parent=None):
        super().__init__(parent)
        
        self.view_name = view_name
        self.model: Optional[ImageModel] = None
        
        # Graphics setup
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # View settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Mouse tracking for tooltips
        self.setMouseTracking(True)
        
        # Pan variables
        self._pan_start = QPointF()
        self._panning = False
    
    def set_model(self, model: ImageModel):
        """Connect to data model."""
        self.model = model
    
    def set_image(self, qimage: QImage):
        """Update displayed image."""
        if qimage.isNull():
            self.pixmap_item.setPixmap(QPixmap())
            return
        key = str(qimage.cacheKey())
        pixmap = QPixmapCache.find(key)
        if pixmap is None:
            pixmap = QPixmap.fromImage(qimage)
            QPixmapCache.insert(key, pixmap)
        self.pixmap_item.setPixmap(pixmap)
        
        # Fit in view on first load
        if self.transform().m11() == 1.0:  # No previous scaling
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming and slice navigation."""
        modifiers = event.modifiers()
        
        if modifiers == Qt.ControlModifier:
            # Zoom with Ctrl+Wheel
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            self.scale(zoom_factor, zoom_factor)
        else:
            # Slice navigation with plain wheel
            delta = 1 if event.angleDelta().y() > 0 else -1
            self.wheelScrolled.emit(delta)
        
        event.accept()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning."""
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement for panning and tooltips."""
        if self._panning:
            # Pan the view
            delta = event.position() - self._pan_start
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            self._pan_start = event.position()
        else:
            # Emit position for tooltip
            scene_pos = self.mapToScene(event.position().toPoint())
            item_pos = self.pixmap_item.mapFromScene(scene_pos)
            
            if self.pixmap_item.contains(item_pos):
                x, y = int(item_pos.x()), int(item_pos.y())
                self.mousePositionChanged.emit(x, y)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        
        super().mouseReleaseEvent(event)
    
    def reset_view(self):
        """Reset zoom and pan to fit image."""
        self.resetTransform()
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)


class MainWindow(QMainWindow):
    """
    Main application window with menu, three-view layout, and controls.
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Medical Image Viewer")
        self.setWindowIcon(QIcon("media/logo.png"))
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)  # Set a good default size
        
        # Create components
        self.setup_ui()
        self.setup_toolbar()
        self.setup_menu()
        self.setup_statusbar()
        
    def setup_ui(self):
        """Create main UI layout."""
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Three-view splitter
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Create three slice views
        self.slice_views = {}
        view_names = ['axial', 'sagittal', 'coronal']
        view_titles = ['Axial (XY)', 'Sagittal (YZ)', 'Coronal (XZ)']
        
        for name, title in zip(view_names, view_titles):
            # View container
            view_frame = QFrame()
            view_frame.setFrameStyle(QFrame.StyledPanel)
            view_layout = QVBoxLayout(view_frame)
            
            # Title label
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; padding: 5px;")
            view_layout.addWidget(title_label)
            
            # Slice view
            slice_view = SliceView(name)
            view_layout.addWidget(slice_view)
            self.slice_views[name] = slice_view
            
            self.splitter.addWidget(view_frame)
        
        # Set proportional sizes (Axial:Sagittal:Coronal = 6:2:1)
        self.splitter.setSizes([600, 200, 200])
        
        # Create control dock
        self.setup_control_dock()
    
    def setup_control_dock(self):
        """Create docked control panel with collapsible sections."""
        self.control_dock = QDockWidget("Controls", self)
        self.control_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_controls_widget = QWidget()
        dock_layout = QVBoxLayout(main_controls_widget)

        # Helper function to create a collapsible group
        def create_collapsible_group(title: str, layout):
            group_box = QGroupBox(title)
            group_box.setCheckable(True)
            
            content_widget = QWidget()
            content_widget.setLayout(layout)
            
            group_box_layout = QVBoxLayout(group_box)
            group_box_layout.setContentsMargins(4, 10, 4, 4)
            group_box_layout.addWidget(content_widget)
            
            group_box.toggled.connect(content_widget.setVisible)
            group_box.setChecked(True)
            return group_box

        # File controls
        file_layout = QVBoxLayout()
        self.load_image_btn = QPushButton("Load Image (Ctrl+O)")
        self.load_labels_btn = QPushButton("Load Labels (Ctrl+L)")
        self.reset_btn = QPushButton("Reset (Ctrl+R)")
        file_layout.addWidget(self.load_image_btn)
        file_layout.addWidget(self.load_labels_btn)
        file_layout.addWidget(self.reset_btn)
        file_layout.addWidget(QLabel("Image Path:"))
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("No image loaded...")
        file_layout.addWidget(self.image_path_input)
        self.update_image_btn = QPushButton("Update Image")
        file_layout.addWidget(self.update_image_btn)
        file_layout.addWidget(QLabel("Label Path:"))
        self.label_path_input = QLineEdit()
        self.label_path_input.setPlaceholderText("No labels loaded...")
        file_layout.addWidget(self.label_path_input)
        self.update_labels_btn = QPushButton("Update Labels")
        file_layout.addWidget(self.update_labels_btn)
        file_group = create_collapsible_group("File Operations", file_layout)
        dock_layout.addWidget(file_group)

        # View controls
        view_layout = QGridLayout()
        self.view_checkboxes = {}
        view_names = ['axial', 'sagittal', 'coronal']
        for i, name in enumerate(view_names):
            checkbox = QCheckBox(name.title())
            checkbox.setChecked(True)
            self.view_checkboxes[name] = checkbox
            view_layout.addWidget(checkbox, 0, i)
        view_group = create_collapsible_group("View Controls", view_layout)
        dock_layout.addWidget(view_group)

        # Slice controls
        self.slice_controls = {}
        for name in view_names:
            slice_layout = QVBoxLayout()
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)
            spinbox = QSpinBox()
            spinbox.setMinimum(0)
            spinbox.setMaximum(100)
            spinbox.setValue(50)
            rotate_btn = QPushButton("Rotate 90°")
            slice_layout.addWidget(QLabel("Slice:"))
            slice_layout.addWidget(slider)
            slice_layout.addWidget(spinbox)
            slice_layout.addWidget(rotate_btn)
            self.slice_controls[name] = {
                'slider': slider, 'spinbox': spinbox, 'rotate_btn': rotate_btn
            }
            slice_group = create_collapsible_group(f"{name.title()} Slice", slice_layout)
            dock_layout.addWidget(slice_group)

        # Overlay controls
        overlay_layout = QVBoxLayout()
        self.global_overlay_cb = QCheckBox("Show Overlay")
        self.global_overlay_cb.setChecked(True)
        self.alpha_label = QLabel("Alpha: 0.50")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        overlay_layout.addWidget(self.global_overlay_cb)
        overlay_layout.addWidget(self.alpha_label)
        overlay_layout.addWidget(self.alpha_slider)
        overlay_group = create_collapsible_group("Overlay Controls", overlay_layout)
        dock_layout.addWidget(overlay_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        dock_layout.addWidget(self.progress_bar)
        
        dock_layout.addStretch()

        scroll_area.setWidget(main_controls_widget)
        self.control_dock.setWidget(scroll_area)
        
        self.control_dock.setMinimumWidth(300)
        self.control_dock.setMaximumWidth(450)
        self.control_dock.resize(320, self.control_dock.height())
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.control_dock)
    
    def setup_toolbar(self):
        """Create toolbar with control panel toggle button."""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        
        # Add some spacing
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)
        
        # Control panel toggle button
        self.panel_toggle_btn = QPushButton("◀")
        self.panel_toggle_btn.setFixedSize(24, 24)
        self.panel_toggle_btn.setToolTip("Hide Control Panel (Ctrl+T)")
        self.panel_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a5a5a;
                border: 1px solid #666;
                border-radius: 12px;
                color: white;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6a6a6a;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """)
        toolbar.addWidget(self.panel_toggle_btn)
        
        # Store toolbar reference
        self.main_toolbar = toolbar
    
    def setup_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_image_action = QAction("Load Image...", self)
        load_image_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_image_action)
        
        load_labels_action = QAction("Load Labels...", self)
        load_labels_action.setShortcut("Ctrl+L")
        file_menu.addAction(load_labels_action)
        
        file_menu.addSeparator()
        
        save_screenshot_action = QAction("Save Screenshot...", self)
        save_screenshot_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_screenshot_action)
        
        file_menu.addSeparator()
        
        reset_action = QAction("Reset", self)
        reset_action.setShortcut("Ctrl+R")
        file_menu.addAction(reset_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        fit_all_action = QAction("Fit All Views", self)
        fit_all_action.setShortcut("F")
        view_menu.addAction(fit_all_action)
        
        view_menu.addSeparator()
        
        toggle_control_panel_action = QAction("Toggle Control Panel", self)
        toggle_control_panel_action.setShortcut("Ctrl+T")
        toggle_control_panel_action.setCheckable(True)
        toggle_control_panel_action.setChecked(True)
        view_menu.addAction(toggle_control_panel_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About...", self)
        about_action.setShortcut("F1")
        help_menu.addAction(about_action)
        
        # Store actions for controller access
        self.actions = {
            'load_image': load_image_action,
            'load_labels': load_labels_action,
            'save_screenshot': save_screenshot_action,
            'reset': reset_action,
            'exit': exit_action,
            'fit_all': fit_all_action,
            'toggle_control_panel': toggle_control_panel_action,
            'about': about_action
        }
    
    def setup_statusbar(self):
        """Create status bar."""
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready - Load an image to begin")
        self.status_bar.addWidget(self.status_label)
        
        # Coordinate display
        self.coord_label = QLabel("Position: -")
        self.status_bar.addPermanentWidget(self.coord_label)


# ============================================================================
# CONTROLLER LAYER - Business Logic & Signal-Slot Binding
# ============================================================================

class ViewerController(QObject):
    """
    Main controller managing interactions between Model and View.
    Handles all user interactions, signal-slot connections, and business logic.
    """
    
    def __init__(self, model: ImageModel, view: MainWindow):
        super().__init__()
        
        self.model = model
        self.view = view
        self.thread_pool = QThreadPool()
        self.default_output_path = None
        
        # Anti-bounce timer for smooth interactions
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_all_views)
        
        # Setup connections
        self.setup_model_connections()
        self.setup_view_connections()
        self.setup_slice_view_connections()
        
        # Connect models to slice views
        for slice_view in self.view.slice_views.values():
            slice_view.set_model(self.model)
    
    def setup_model_connections(self):
        """Connect model signals."""
        self.model.imageLoaded.connect(self._on_image_loaded)
        self.model.labelLoaded.connect(self._on_labels_loaded)
        self.model.loadError.connect(self._on_load_error)
        self.model.loadProgress.connect(self._on_load_progress)
    
    def setup_view_connections(self):
        """Connect main view signals."""
        # Menu actions
        self.view.actions['load_image'].triggered.connect(self.load_image)
        self.view.actions['load_labels'].triggered.connect(self.load_labels)
        self.view.actions['reset'].triggered.connect(self.reset_all)
        self.view.actions['exit'].triggered.connect(self.view.close)
        self.view.actions['fit_all'].triggered.connect(self.fit_all_views)
        self.view.actions['save_screenshot'].triggered.connect(self.save_screenshot)
        self.view.actions['toggle_control_panel'].triggered.connect(self.toggle_control_panel)
        self.view.actions['about'].triggered.connect(self.show_about)
        
        # Control buttons
        self.view.load_image_btn.clicked.connect(self.load_image)
        self.view.load_labels_btn.clicked.connect(self.load_labels)
        self.view.reset_btn.clicked.connect(self.reset_all)
        
        # Path input buttons
        self.view.update_image_btn.clicked.connect(self.update_image_from_path)
        self.view.update_labels_btn.clicked.connect(self.update_labels_from_path)
        
        # View visibility
        for name, checkbox in self.view.view_checkboxes.items():
            checkbox.toggled.connect(lambda checked, n=name: self._toggle_view_visibility(n, checked))
        
        # Slice controls
        for name, controls in self.view.slice_controls.items():
            slider = controls['slider']
            spinbox = controls['spinbox']
            rotate_btn = controls['rotate_btn']
            
            slider.valueChanged.connect(lambda v, n=name: self._on_slice_changed(n, v))
            spinbox.valueChanged.connect(lambda v, n=name: self._on_slice_changed(n, v))
            rotate_btn.clicked.connect(lambda checked, n=name: self._rotate_view(n))
        
        # Overlay controls
        self.view.global_overlay_cb.toggled.connect(self._on_global_overlay_toggled)
        self.view.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        
        # Control panel toggle button
        self.view.panel_toggle_btn.clicked.connect(self.toggle_control_panel)
    
    def setup_slice_view_connections(self):
        """Connect slice view signals."""
        for name, slice_view in self.view.slice_views.items():
            slice_view.wheelScrolled.connect(lambda delta, n=name: self._on_wheel_scroll(n, delta))
            slice_view.mousePositionChanged.connect(lambda x, y, n=name: self._on_mouse_position(n, x, y))
    
    def load_image(self):
        """Load image file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self.view,
            "Load Medical Image",
            "",
            "Image Files (*.nii *.nii.gz *.mha *.mhd);;All Files (*)"
        )
        
        if filepath:
            self.view.status_label.setText("Loading image...")

            # Show progress bar if control panel exists
            if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
                self.view.progress_bar.setValue(0)
                self.view.progress_bar.setVisible(True)

            # Load in background
            worker = LoadWorker(self.model, filepath, False)
            self.thread_pool.start(worker)
    
    def load_labels(self):
        """Load labels file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self.view,
            "Load Medical Labels",
            "",
            "Label Files (*.nii *.nii.gz *.mha *.mhd);;All Files (*)"
        )
        
        if filepath:
            self.view.status_label.setText("Loading labels...")

            # Show progress bar if control panel exists
            if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
                self.view.progress_bar.setValue(0)
                self.view.progress_bar.setVisible(True)

            # Load in background
            worker = LoadWorker(self.model, filepath, True)
            self.thread_pool.start(worker)
    
    def update_image_from_path(self):
        """Load image from path input field."""
        # Check if control panel exists
        if not (hasattr(self.view, 'control_dock') and self.view.control_dock is not None):
            return
            
        filepath = self.view.image_path_input.text().strip()
        if not filepath:
            QMessageBox.warning(self.view, "Warning", "Please enter an image file path")
            return
        
        if not Path(filepath).exists():
            QMessageBox.warning(self.view, "Warning", f"File not found: {filepath}")
            return
        
        self.view.status_label.setText("Loading image from path...")
        self.view.progress_bar.setValue(0)
        self.view.progress_bar.setVisible(True)
        
        # Load in background
        worker = LoadWorker(self.model, filepath, False)
        self.thread_pool.start(worker)
    
    def update_labels_from_path(self):
        """Load labels from path input field."""
        # Check if control panel exists
        if not (hasattr(self.view, 'control_dock') and self.view.control_dock is not None):
            return
            
        filepath = self.view.label_path_input.text().strip()
        if not filepath:
            QMessageBox.warning(self.view, "Warning", "Please enter a label file path")
            return
        
        if not Path(filepath).exists():
            QMessageBox.warning(self.view, "Warning", f"File not found: {filepath}")
            return
        
        self.view.status_label.setText("Loading labels from path...")
        self.view.progress_bar.setValue(0)
        self.view.progress_bar.setVisible(True)
        
        # Load in background
        worker = LoadWorker(self.model, filepath, True)
        self.thread_pool.start(worker)
    
    def reset_all(self):
        """Reset all views and data."""
        # Clear model
        self.model.image_data = None
        self.model.label_data = None
        self.model.get_slice_data.cache_clear()
        self.model._render_slice_cached.cache_clear()
        QPixmapCache.clear()
        
        # Reset UI - check if control panel exists
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            for controls in self.view.slice_controls.values():
                controls['slider'].setValue(0)
                controls['spinbox'].setValue(0)
            
            # Clear path inputs
            self.view.image_path_input.clear()
            self.view.label_path_input.clear()
        
        for slice_view in self.view.slice_views.values():
            slice_view.scene.clear()
            slice_view.pixmap_item = QGraphicsPixmapItem()
            slice_view.scene.addItem(slice_view.pixmap_item)
        
        self.view.status_label.setText("Ready - Load an image to begin")
        
        # Reset progress bar if control panel exists
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)
    
    def fit_all_views(self):
        """Fit all views to show full image."""
        for slice_view in self.view.slice_views.values():
            slice_view.reset_view()
    
    def save_screenshot(self):
        """Save current view as screenshot."""
        default_name = self.default_output_path or "screenshot.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self.view,
            "Save Screenshot",
            default_name,
            "PNG Files (*.png);;All Files (*)"
        )
        
        if filepath:
            # Capture the central widget
            pixmap = self.view.centralWidget().grab()
            pixmap.save(filepath)
            self.view.status_label.setText(f"Screenshot saved: {Path(filepath).name}")
    
    def show_about(self):
        """Show About dialog."""
        about_dialog = AboutDialog(self.view)
        about_dialog.exec()
    
    def toggle_control_panel(self):
        """Toggle control panel visibility - completely remove/add dock."""
        is_visible = hasattr(self.view, 'control_dock') and self.view.control_dock is not None
        
        if is_visible:
            # Store current window and dock sizes
            current_size = self.view.size()
            dock_width = self.view.control_dock.width()
            self.view._stored_dock_width = dock_width
            self.view._stored_window_width = current_size.width()
            
            # Remove and destroy the control dock
            self.view.removeDockWidget(self.view.control_dock)
            self.view.control_dock.deleteLater()
            self.view.control_dock = None
            
            # Make window narrower by removing the dock width
            new_width = max(current_size.width() - dock_width - 10, 600)  # Minimum 600px
            self.view.resize(new_width, current_size.height())
            
            # Update button appearance
            self.view.panel_toggle_btn.setText("▶")
            self.view.panel_toggle_btn.setToolTip("Show Control Panel (Ctrl+T)")
            self.view.actions['toggle_control_panel'].setChecked(False)
            
        else:
            # Recreate the control dock
            self.view.setup_control_dock()
            
            # Reconnect the control panel signals
            self._reconnect_control_signals()
            
            # Restore window width
            if hasattr(self.view, '_stored_window_width'):
                current_size = self.view.size()
                self.view.resize(self.view._stored_window_width, current_size.height())
            else:
                # Default expanded width
                current_size = self.view.size()
                self.view.resize(current_size.width() + 300, current_size.height())
            
            # Update button appearance
            self.view.panel_toggle_btn.setText("◀")
            self.view.panel_toggle_btn.setToolTip("Hide Control Panel (Ctrl+T)")
            self.view.actions['toggle_control_panel'].setChecked(True)
    
    def _reconnect_control_signals(self):
        """Reconnect control panel signals after recreating the dock."""
        # Control buttons
        self.view.load_image_btn.clicked.connect(self.load_image)
        self.view.load_labels_btn.clicked.connect(self.load_labels)
        self.view.reset_btn.clicked.connect(self.reset_all)
        
        # Path input buttons
        self.view.update_image_btn.clicked.connect(self.update_image_from_path)
        self.view.update_labels_btn.clicked.connect(self.update_labels_from_path)
        
        # View visibility
        for name, checkbox in self.view.view_checkboxes.items():
            checkbox.toggled.connect(lambda checked, n=name: self._toggle_view_visibility(n, checked))
        
        # Slice controls
        for name, controls in self.view.slice_controls.items():
            slider = controls['slider']
            spinbox = controls['spinbox']
            rotate_btn = controls['rotate_btn']
            
            slider.valueChanged.connect(lambda v, n=name: self._on_slice_changed(n, v))
            spinbox.valueChanged.connect(lambda v, n=name: self._on_slice_changed(n, v))
            rotate_btn.clicked.connect(lambda checked, n=name: self._rotate_view(n))
        
        # Overlay controls
        self.view.global_overlay_cb.toggled.connect(self._on_global_overlay_toggled)
        self.view.alpha_slider.valueChanged.connect(self._on_alpha_changed)
    
    def _on_image_loaded(self, filepath: str, shape: tuple):
        """Handle successful image loading."""
        self.view.status_label.setText(f"Image loaded: {Path(filepath).name} - Shape: {shape}")
        
        # Update control panel elements if they exist
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)
            
            # Update image path input
            self.view.image_path_input.setText(filepath)
            
            # Update slice controls
            for name, config in self.model.view_configs.items():
                axis = config['axis']
                max_slice = shape[axis] - 1
                mid_slice = max_slice // 2
                
                controls = self.view.slice_controls[name]
                controls['slider'].setMaximum(max_slice)
                controls['spinbox'].setMaximum(max_slice)
                controls['slider'].setValue(mid_slice)
                controls['spinbox'].setValue(mid_slice)
                
                self.model.set_slice(name, mid_slice)
        else:
            # Set model slice positions directly when no controls exist
            for name, config in self.model.view_configs.items():
                axis = config['axis']
                mid_slice = shape[axis] // 2
                self.model.set_slice(name, mid_slice)
        
        self._update_all_views()
    
    def _on_labels_loaded(self, filepath: str, unique_count: int):
        """Handle successful label loading."""
        self.view.status_label.setText(f"Labels loaded: {Path(filepath).name} - {unique_count} unique labels")
        
        # Update control panel elements if they exist
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)
            # Update label path input
            self.view.label_path_input.setText(filepath)
        
        self._update_all_views()
    
    def _on_load_error(self, error_msg: str):
        """Handle loading errors."""
        self.view.status_label.setText("Load failed")
        
        # Hide progress bar if control panel exists
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)

        QMessageBox.critical(self.view, "Load Error", error_msg)

    def _on_load_progress(self, value: int):
        """Update progress bar during loading."""
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setValue(value)
            if value >= 100:
                self.view.progress_bar.setVisible(False)
    
    def _toggle_view_visibility(self, view_name: str, visible: bool):
        """Toggle view visibility."""
        self.model.view_configs[view_name]['show'] = visible
        
        # Find the view frame and hide/show it
        view_index = ['axial', 'sagittal', 'coronal'].index(view_name)
        widget = self.view.splitter.widget(view_index)
        widget.setVisible(visible)
    
    def _on_slice_changed(self, view_name: str, value: int):
        """Handle slice navigation."""
        # Sync slider and spinbox
        controls = self.view.slice_controls[view_name]
        controls['slider'].blockSignals(True)
        controls['spinbox'].blockSignals(True)
        controls['slider'].setValue(value)
        controls['spinbox'].setValue(value)
        controls['slider'].blockSignals(False)
        controls['spinbox'].blockSignals(False)
        
        # Update model
        self.model.set_slice(view_name, value)
        
        # Debounced update
        self.update_timer.start(40)  # 40ms debounce for smooth scrolling
    
    def _on_wheel_scroll(self, view_name: str, delta: int):
        """Handle mouse wheel slice navigation."""
        controls = self.view.slice_controls[view_name]
        current_value = controls['slider'].value()
        new_value = current_value + delta
        
        # Clamp to valid range
        max_value = controls['slider'].maximum()
        new_value = max(0, min(new_value, max_value))
        
        self._on_slice_changed(view_name, new_value)
    
    def _on_mouse_position(self, view_name: str, x: int, y: int):
        """Handle mouse position for tooltips and coordinates."""
        # Update coordinate display
        self.view.coord_label.setText(f"Position: ({x}, {y})")
        
        # Show label tooltip if available
        if self.model.label_data is not None:
            label_value = self.model.get_label_at_position(view_name, x, y)
            if label_value > 0:
                QToolTip.showText(
                    self.view.slice_views[view_name].mapToGlobal(self.view.slice_views[view_name].mapFromScene(QPointF(x, y))),
                    f"Label: {label_value}"
                )
    
    def _rotate_view(self, view_name: str):
        """Rotate view by 90 degrees."""
        config = self.model.view_configs[view_name]
        config['rotation'] = (config['rotation'] + 90) % 360
        self._update_view(view_name)
    
    def _on_global_overlay_toggled(self, checked: bool):
        """Handle global overlay toggle."""
        self.model.global_overlay = checked
        self._update_all_views()
    
    def _on_alpha_changed(self, value: int):
        """Handle alpha slider changes."""
        alpha = value / 100.0
        self.model.global_alpha = alpha
        self.view.alpha_label.setText(f"Alpha: {alpha:.2f}")
        
        # Update all view configs
        for config in self.model.view_configs.values():
            config['alpha'] = alpha
        
        self._update_all_views()
    
    def _update_all_views(self):
        """Update all visible slice views."""
        for name, config in self.model.view_configs.items():
            if config['show']:
                self._update_view(name)
    
    def _update_view(self, view_name: str):
        """Update a specific view."""
        if self.model.image_data is None:
            return
        
        config = self.model.view_configs[view_name]
        slice_idx = config['slice']
        show_overlay = config['overlay'] and self.model.global_overlay
        alpha = config['alpha']
        
        # Render slice
        qimage = self.model.render_slice(view_name, slice_idx, show_overlay, alpha)
        
        # Update view
        slice_view = self.view.slice_views[view_name]
        slice_view.set_image(qimage)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="nifti_viewer_qt",
        description="Interactive medical image viewer: three-view display, label overlay, zoomable and rotatable.",
        epilog="""
Keyboard Shortcuts:
  Ctrl+O     Load image file
  Ctrl+L     Load label file  
  Ctrl+S     Save current view as PNG
  Ctrl+R     Reset views and cache
  Ctrl+T     Toggle control panel visibility
  Ctrl+Wheel Zoom view
  Wheel      Navigate slices
  Right-Drag Pan view
  F          Fit all views
  F1         Show about dialog
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--image", help="Load image file directly (.nii, .nii.gz, .mha, .mhd)")
    parser.add_argument("-l", "--label", help="Load label file directly (.nii, .nii.gz, .mha, .mhd)")
    parser.add_argument("-o", "--output", help="Default path for screenshot saving")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s 0.1.0")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help="Set logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Medical Image Viewer")
    app.setApplicationVersion("0.1.0")
    
    # Set application style
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QDockWidget {
            background-color: #3c3c3c;
            color: white;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555;
            border-radius: 5px;
            margin-top: 1ex;
            color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QGroupBox::indicator {
            width: 18px;
            height: 18px;
            right: 5px;
        }
        QGroupBox::indicator:unchecked {
            image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwYXRoIGQ9Ik0wIDBoMjR2MjRIMDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTEwIDE3bDUtNS01LTV2MTB6Ii8+PC9zdmc+);
        }
        QGroupBox::indicator:checked {
            image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwYXRoIGQ9Ik0wIDBoMjR2MjRIMDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTcgMTBsNSA1IDUtNXoiLz48L3N2Zz4=);
        }
        QPushButton {
            background-color: #4a4a4a;
            border: 1px solid #666;
            padding: 5px;
            border-radius: 3px;
            color: white;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QPushButton:pressed {
            background-color: #3a3a3a;
        }
        QSlider::groove:horizontal {
            border: 1px solid #666;
            height: 8px;
            background: #2b2b2b;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #4a90e2;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 3px;
        }
        QCheckBox {
            color: white;
        }
        QLabel {
            color: white;
        }
        QSpinBox {
            background-color: #4a4a4a;
            border: 1px solid #666;
            padding: 2px;
            color: white;
        }
        QLineEdit {
            background-color: #4a4a4a;
            border: 1px solid #666;
            padding: 4px;
            border-radius: 3px;
            color: white;
            selection-background-color: #4a90e2;
        }
        QLineEdit:focus {
            border: 2px solid #4a90e2;
        }
        QStatusBar {
            background-color: #3c3c3c;
            color: white;
        }
    """)
    
    try:
        # Create MVC components
        model = ImageModel()
        view = MainWindow()
        controller = ViewerController(model, view)
        
        # Handle command line arguments
        if args.image:
            if Path(args.image).exists():
                model.load_image(args.image)
            else:
                logging.warning(f"Image file not found: {args.image}")
                
        if args.label:
            if Path(args.label).exists():
                model.load_labels(args.label)
            else:
                logging.warning(f"Label file not found: {args.label}")
        
        # Store output path for screenshots
        if args.output:
            controller.default_output_path = args.output
        
        # Show main window
        view.show()
        
        # Run application
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        logging.error(traceback.format_exc())
        QMessageBox.critical(None, "Application Error", 
                           f"Failed to start application:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
