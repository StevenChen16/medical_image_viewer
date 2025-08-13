#!/usr/bin/env python3
"""
NIfTI/MHA Viewer with PySide 6
A medical image viewer for NIfTI and MHA format files.
"""

import time
import dataclasses
import csv
__version__ = "0.1.4"
__author__ = "Steven Chen"
__license__ = "MIT"
__copyright__ = "Copyright 2025, Steven Chen"
__all__ = ["MainWindow", "ViewerController", "ImageModel", "main"]

import argparse
import logging
import os
import sys
import traceback
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image

# Conditionally import SimpleITK for MHA support
try:
    import SimpleITK as sitk

    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

from PySide6.QtCore import (QLine, QLineF, QObject, QPoint, QPointF, QRect,
                            QRectF, QRunnable, QSize, Qt, QThread, QThreadPool,
                            QTimer, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QBrush, QColor, QCursor, QFont,
                           QFontMetricsF, QIcon, QImage, QMouseEvent, QPainter,
                           QPainterPath, QPen, QPixmap, QPixmapCache,
                           QShortcut, QTransform, QWheelEvent)
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog,
                               QComboBox, QDialog, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFrame, QGraphicsEllipseItem,
                               QGraphicsItem, QGraphicsLineItem,
                               QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMenuBar,
                               QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QSplitter, QStatusBar, QTabWidget, QTextEdit,
                               QToolTip, QVBoxLayout, QWidget)

# ============================================================================
# INTERNATIONALIZATION SYSTEM
# ============================================================================


class TranslationManager:
    """Centralized translation management for multilingual support."""
    
    def __init__(self):
        self.current_language = "en"  # Default to English
        self.translations = {
            "en": {
                # Window titles
                "window_title": "Medical Image Viewer",
                "about_title": "About Medical Image Viewer",
                
                # Menu items
                "menu_file": "File",
                "menu_view": "View",
                "menu_help": "Help",
                "load_image": "Load Image...",
                "load_labels": "Load Labels...",
                "save": "Save",
                "save_image": "Image Only...",
                "save_label": "Label Only...",
                "save_overlay": "Overlay Image...",
                "save_screenshot": "Save Screenshot...",
                "reset": "Reset",
                "exit": "Exit",
                "fit_all_views": "Fit All Views",
                "toggle_control_panel": "Toggle Control Panel",
                "about": "About...",
                "language": "Language",
                
                # Status messages
                "status_ready": "Ready - Load an image to begin",
                "status_loading_image": "Loading image...",
                "status_loading_labels": "Loading labels...",
                "status_loading_image_path": "Loading image from path...",
                "status_loading_labels_path": "Loading labels from path...",
                "status_load_failed": "Load failed",
                "position": "Position:",
                
                # Control panel
                "controls": "Controls",
                "load_image_btn": "Load Image (Ctrl+O)",
                "load_labels_btn": "Load Labels (Ctrl+L)",
                "reset_btn": "Reset (Ctrl+R)",
                "image_path": "Image Path:",
                "update_image": "Update Image",
                "label_path": "Label Path:",
                "update_labels": "Update Labels",
                "overlay_settings": "Overlay Settings",
                "show_overlay": "Show Overlay",
                "alpha": "Alpha:",
                "label_colors": "Label Colors",
                "no_labels_found": "No labels found",
                "no_labels_loaded": "No labels loaded",
                "label_prefix": "Label",
                
                # View titles
                "axial_view": "Axial (XY)",
                "sagittal_view": "Sagittal (YZ)",
                "coronal_view": "Coronal (XZ)",
                
                # About dialog
                "close": "Close",
                "english": "English",
                "chinese": "中文",
                "french": "Français",
                
                # Control panel groups
                "file_operations": "File Operations",
                "view_controls": "View Controls",
                "overlay_controls": "Overlay Controls",
                "measurement_tools": "Measurement Tools",
                "measurement_settings": "Measurement Settings",
                
                # Control panel elements
                "no_image_loaded": "No image loaded...",
                "no_labels_loaded_placeholder": "No labels loaded...",
                "slice": "Slice:",
                "rotate_90": "Rotate 90°",
                "reset_to_defaults": "Reset to Defaults",
                "alpha_label": "Alpha: 0.50",
                
                # View names
                "axial": "Axial",
                "sagittal": "Sagittal", 
                "coronal": "Coronal",
                
                # Measurement tools
                "line_tool": "Line Tool",
                "eraser": "Eraser",
                "export_csv": "Export CSV",
                "measurements": "Measurements:",
                "select_all": "Select All",
                "deselect_all": "Deselect All",
                "line_width": "Line Width:",
                "line_color": "Line Color:",
                "text_color": "Text Color:",
                "font_size": "Font Size:",
                "font_weight": "Font Weight:",
                "auto_snap": "Auto-Snap:",
                "enable_endpoint_snapping": "Enable endpoint snapping",
                "snap_distance": "Snap Distance:",
                "apply_to_selected": "Apply to Selected",
                "apply_to_all": "Apply to All",
                "normal": "Normal",
                "bold": "Bold",
            },
            "zh": {
                # Window titles
                "window_title": "医学影像查看器",
                "about_title": "关于医学影像查看器",
                
                # Menu items
                "menu_file": "文件",
                "menu_view": "视图",
                "menu_help": "帮助",
                "load_image": "加载影像...",
                "load_labels": "加载标签...",
                "save": "保存",
                "save_image": "仅影像...",
                "save_label": "仅标签...",
                "save_overlay": "叠加影像...",
                "save_screenshot": "保存截图...",
                "reset": "重置",
                "exit": "退出",
                "fit_all_views": "适应所有视图",
                "toggle_control_panel": "切换控制面板",
                "about": "关于...",
                "language": "语言",
                
                # Status messages
                "status_ready": "就绪 - 加载影像开始使用",
                "status_loading_image": "正在加载影像...",
                "status_loading_labels": "正在加载标签...",
                "status_loading_image_path": "正在从路径加载影像...",
                "status_loading_labels_path": "正在从路径加载标签...",
                "status_load_failed": "加载失败",
                "position": "位置:",
                
                # Control panel
                "controls": "控制",
                "load_image_btn": "加载影像 (Ctrl+O)",
                "load_labels_btn": "加载标签 (Ctrl+L)",
                "reset_btn": "重置 (Ctrl+R)",
                "image_path": "影像路径:",
                "update_image": "更新影像",
                "label_path": "标签路径:",
                "update_labels": "更新标签",
                "overlay_settings": "叠加设置",
                "show_overlay": "显示叠加",
                "alpha": "透明度:",
                "label_colors": "标签颜色",
                "no_labels_found": "未找到标签",
                "no_labels_loaded": "未加载标签",
                "label_prefix": "标签",
                
                # View titles
                "axial_view": "轴向 (XY)",
                "sagittal_view": "矢状 (YZ)",
                "coronal_view": "冠状 (XZ)",
                
                # About dialog
                "close": "关闭",
                "english": "English",
                "chinese": "中文",
                "french": "Français",
                
                # Control panel groups
                "file_operations": "文件操作",
                "view_controls": "视图控制",
                "overlay_controls": "叠加控制",
                "measurement_tools": "测量工具",
                "measurement_settings": "测量设置",
                
                # Control panel elements
                "no_image_loaded": "未加载影像...",
                "no_labels_loaded_placeholder": "未加载标签...",
                "slice": "切片:",
                "rotate_90": "旋转90°",
                "reset_to_defaults": "重置为默认",
                "alpha_label": "透明度: 0.50",
                
                # View names
                "axial": "轴向",
                "sagittal": "矢状",
                "coronal": "冠状",
                
                # Measurement tools
                "line_tool": "线条工具",
                "eraser": "橡皮擦",
                "export_csv": "导出CSV",
                "measurements": "测量:",
                "select_all": "全选",
                "deselect_all": "取消全选",
                "line_width": "线宽:",
                "line_color": "线条颜色:",
                "text_color": "文字颜色:",
                "font_size": "字体大小:",
                "font_weight": "字体粗细:",
                "auto_snap": "自动吸附:",
                "enable_endpoint_snapping": "启用端点吸附",
                "snap_distance": "吸附距离:",
                "apply_to_selected": "应用到选中",
                "apply_to_all": "应用到全部",
                "normal": "正常",
                "bold": "粗体",
            },
            "fr": {
                # Window titles
                "window_title": "Visionneuse d'Images Médicales",
                "about_title": "À propos de la Visionneuse d'Images Médicales",
                
                # Menu items
                "menu_file": "Fichier",
                "menu_view": "Affichage",
                "menu_help": "Aide",
                "load_image": "Charger une Image...",
                "load_labels": "Charger des Étiquettes...",
                "save": "Enregistrer",
                "save_image": "Image Seule...",
                "save_label": "Étiquette Seule...",
                "save_overlay": "Image Superposée...",
                "save_screenshot": "Enregistrer la Capture d'Écran...",
                "reset": "Réinitialiser",
                "exit": "Quitter",
                "fit_all_views": "Ajuster Toutes les Vues",
                "toggle_control_panel": "Basculer le Panneau de Contrôle",
                "about": "À propos...",
                "language": "Langue",
                
                # Status messages
                "status_ready": "Prêt - Chargez une image pour commencer",
                "status_loading_image": "Chargement de l'image...",
                "status_loading_labels": "Chargement des étiquettes...",
                "status_loading_image_path": "Chargement de l'image depuis le chemin...",
                "status_loading_labels_path": "Chargement des étiquettes depuis le chemin...",
                "status_load_failed": "Échec du chargement",
                "position": "Position:",
                
                # Control panel
                "controls": "Contrôles",
                "load_image_btn": "Charger une Image (Ctrl+O)",
                "load_labels_btn": "Charger des Étiquettes (Ctrl+L)",
                "reset_btn": "Réinitialiser (Ctrl+R)",
                "image_path": "Chemin de l'Image:",
                "update_image": "Mettre à jour l'Image",
                "label_path": "Chemin des Étiquettes:",
                "update_labels": "Mettre à jour les Étiquettes",
                "overlay_settings": "Paramètres de Superposition",
                "show_overlay": "Afficher la Superposition",
                "alpha": "Alpha:",
                "label_colors": "Couleurs des Étiquettes",
                "no_labels_found": "Aucune étiquette trouvée",
                "no_labels_loaded": "Aucune étiquette chargée",
                "label_prefix": "Étiquette",
                
                # View titles
                "axial_view": "Axial (XY)",
                "sagittal_view": "Sagittal (YZ)",
                "coronal_view": "Coronal (XZ)",
                
                # About dialog
                "close": "Fermer",
                "english": "English",
                "chinese": "中文",
                "french": "Français",
                
                # Control panel groups
                "file_operations": "Opérations sur les Fichiers",
                "view_controls": "Contrôles d'Affichage",
                "overlay_controls": "Contrôles de Superposition",
                "measurement_tools": "Outils de Mesure",
                "measurement_settings": "Paramètres de Mesure",
                
                # Control panel elements
                "no_image_loaded": "Aucune image chargée...",
                "no_labels_loaded_placeholder": "Aucune étiquette chargée...",
                "slice": "Tranche:",
                "rotate_90": "Rotation 90°",
                "reset_to_defaults": "Rétablir par Défaut",
                "alpha_label": "Alpha: 0.50",
                
                # View names
                "axial": "Axial",
                "sagittal": "Sagittal",
                "coronal": "Coronal",
                
                # Measurement tools
                "line_tool": "Outil Ligne",
                "eraser": "Gomme",
                "export_csv": "Exporter CSV",
                "measurements": "Mesures:",
                "select_all": "Tout Sélectionner",
                "deselect_all": "Tout Désélectionner",
                "line_width": "Largeur de Ligne:",
                "line_color": "Couleur de Ligne:",
                "text_color": "Couleur du Texte:",
                "font_size": "Taille de Police:",
                "font_weight": "Graisse de Police:",
                "auto_snap": "Accrochage Auto:",
                "enable_endpoint_snapping": "Activer l'accrochage des points",
                "snap_distance": "Distance d'Accrochage:",
                "apply_to_selected": "Appliquer à la Sélection",
                "apply_to_all": "Appliquer à Tout",
                "normal": "Normal",
                "bold": "Gras",
            }
        }
    
    def set_language(self, language_code: str) -> None:
        """Set the current language."""
        if language_code in self.translations:
            self.current_language = language_code
    
    def get_language(self) -> str:
        """Get the current language code."""
        return self.current_language
    
    def get_text(self, key: str) -> str:
        """Get translated text for the given key."""
        return self.translations.get(self.current_language, {}).get(key, key)
    
    def get_available_languages(self) -> list:
        """Get list of available language codes."""
        return list(self.translations.keys())


# Global translation manager instance
_translation_manager = TranslationManager()


def tr(key: str) -> str:
    """Translate text using the global translation manager."""
    return _translation_manager.get_text(key)


def set_language(language_code: str) -> None:
    """Set the global application language."""
    _translation_manager.set_language(language_code)


def get_current_language() -> str:
    """Get the current application language."""
    return _translation_manager.get_language()


# ============================================================================
# UTILITY WIDGETS
# ============================================================================


class ColorButton(QPushButton):
    """Custom color picker button widget."""

    colorChanged = Signal(object)  # Emits QColor when color changes

    def __init__(self, *args, color=None, **kwargs):
        super().__init__(*args, **kwargs)
        if color is None:
            # Default to first enhanced color (red)
            self._color = QColor(227, 26, 28)
        elif isinstance(color, (list, tuple)) and len(color) == 3:
            self._color = QColor(*color)  # Unpack RGB tuple
        elif isinstance(color, QColor):
            self._color = color
        else:
            self._color = QColor(color)
        self.pressed.connect(self.open_color_picker)
        self.setFixedSize(30, 24)
        self.update_button_color()

    def setColor(self, color):
        """Set button color and emit signal if changed."""
        if isinstance(color, (list, tuple)) and len(color) == 3:
            color = QColor(*color)
        elif not isinstance(color, QColor):
            color = QColor(color)

        if color != self._color:
            self._color = color
            self.update_button_color()
            self.colorChanged.emit(self._color)

    def color(self):
        """Get current color as QColor."""
        return self._color

    def color_rgb(self):
        """Get current color as RGB tuple."""
        return (self._color.red(), self._color.green(), self._color.blue())

    def update_button_color(self):
        """Update button visual appearance to show current color."""
        if self._color.isValid():
            self.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: rgb({self._color.red()}, {self._color.green()}, {self._color.blue()});
                    border: 2px solid #666;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    border: 2px solid #999;
                }}
                QPushButton:pressed {{
                    border: 2px solid #ccc;
                }}
            """
            )
        else:
            self.setStyleSheet("")

    def open_color_picker(self):
        """Open color picker dialog."""
        dialog = QColorDialog(self._color, self)
        dialog.setWindowTitle("Select Label Color")

        if dialog.exec() == QColorDialog.Accepted:
            new_color = dialog.currentColor()
            if new_color.isValid():
                self.setColor(new_color)


# ============================================================================
# MODEL LAYER - Data Management & Caching
# ============================================================================


class ImageModel(QObject):
    """
    Core data model for medical image and label management.
    Handles loading, caching, and vectorized rendering for NIfTI and MHA.
    """

    # Signals
    # filename, shape, affine, spacing
    imageLoaded = Signal(str, tuple, object, object)
    labelLoaded = Signal(str, int)  # filename, unique_labels_count
    sliceReady = Signal(str, int, QImage)  # view_name, slice_idx, image
    loadError = Signal(str)  # error_message
    loadProgress = Signal(int)  # loading progress percentage

    def __init__(self, pixmap_cache_size: int = 102400):  # 100MB default
        super().__init__()

        # Core data storage
        self.image_data: np.ndarray | None = None
        self.label_data: np.ndarray | None = None
        self.image_affine: np.ndarray | None = None
        self.image_spacing: tuple[float, float, float] | None = None

        # File paths
        self.image_path: str = ""
        self.label_path: str = ""

        # View configurations
        self.view_configs = {
            "axial": {
                "axis": 2,
                "slice": 0,
                "show": True,
                "overlay": True,
                "rotation": 0,
                "alpha": 0.5,
            },
            "sagittal": {
                "axis": 0,
                "slice": 0,
                "show": True,
                "overlay": True,
                "rotation": 0,
                "alpha": 0.5,
            },
            "coronal": {
                "axis": 1,
                "slice": 0,
                "show": True,
                "overlay": True,
                "rotation": 0,
                "alpha": 0.5,
            },
        }

        # Global settings
        self.global_overlay = True
        self.global_alpha = 0.5

        # Medical-optimized ColorBrewer Paired palette - reordered for maximum contrast
        # Alternates between different hue families for better medical visualization
        # Original ColorBrewer colors maintained but resequenced for consecutive contrast
        self._default_colors = [
            (227, 26, 28),  # 1. Red - high contrast, medically significant
            (178, 223, 138),  # 2. Light green - strong contrast to red
            (31, 120, 180),  # 3. Dark blue - distinct from green
            (253, 191, 111),  # 4. Orange - warm contrast to blue
            (106, 61, 154),  # 5. Purple - cool contrast to orange
            (255, 255, 153),  # 6. Yellow - bright contrast to purple
            (166, 206, 227),  # 7. Light blue - different from dark blue above
            (177, 89, 40),  # 8. Brown - earth tone contrast
            (251, 154, 153),  # 9. Light red - softer than primary red
            (51, 160, 44),  # 10. Dark green - different from light green
            (255, 127, 0),  # 11. Dark orange - vibrant accent
            (202, 178, 214),  # 12. Light purple - subtle final color
        ]

        # Dynamic color mapping: label_value -> (r, g, b) tuple
        self._custom_label_colors = {}  # User customizations
        self._current_label_values = []  # Currently loaded label values

        # Fixed overlay transparency for clinical consistency
        self._overlay_alpha = 0.4

        # Configure QPixmapCache
        QPixmapCache.setCacheLimit(pixmap_cache_size)

    def _load_image_data(self, filepath: str) -> tuple[np.ndarray, np.ndarray, tuple]:
        """
        Internal helper to load medical image data from various formats.
        Returns a tuple of (NumPy array, affine matrix, voxel spacing).
        """
        file_ext = "".join(Path(filepath).suffixes).lower()

        if file_ext in [".nii", ".nii.gz"]:
            img = nib.load(filepath)
            affine = img.affine
            spacing = img.header.get_zooms()
            return img.get_fdata(dtype=np.float32), affine, spacing

        elif file_ext in [".mha", ".mhd"]:
            if not SITK_AVAILABLE:
                raise ImportError(
                    "SimpleITK is required to load MHA/MHD files. Please run: pip install SimpleITK"
                )

            itk_img = sitk.ReadImage(filepath)
            spacing = itk_img.GetSpacing()
            origin = itk_img.GetOrigin()

            # Create a simplified affine matrix from spacing and origin
            affine = np.eye(4)
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]
            affine[0, 3] = origin[0]
            affine[1, 3] = origin[1]
            affine[2, 3] = origin[2]

            # Transpose from ITK's (Z, Y, X) to nibabel's (X, Y, Z) convention
            np_array = sitk.GetArrayFromImage(itk_img)
            return np_array.transpose(2, 1, 0).astype(np.float32), affine, spacing

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @lru_cache(maxsize=50)
    def get_slice_data(
        self, axis: int, slice_idx: int, is_label: bool = False
    ) -> np.ndarray:
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
            normalized = ((img_slice - img_min) / (img_max - img_min) * 255).astype(
                np.uint8
            )
        else:
            normalized = np.zeros_like(img_slice, dtype=np.uint8)
        return normalized

    def create_label_overlay(self, label_slice: np.ndarray) -> np.ndarray | None:
        """
        Create medical-grade colored overlay from label slice.
        Uses ColorBrewer Paired palette with user customizations support.

        Args:
            label_slice: 2D integer array, labels start from 1; 0 = no label
        Returns:
            H×W×3 uint8 pseudocolor overlay, or None if no labels
        """
        if label_slice.size == 0:
            return None

        # Ensure proper integer type for indexing
        labels = label_slice.astype(np.intp, copy=False)
        mask = labels > 0
        if not np.any(mask):
            return None

        # Create overlay using dynamic color mapping
        h, w = label_slice.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Get unique label values in this slice (excluding background)
        unique_labels = np.unique(labels[mask])

        # Apply colors for each label using vectorized operations
        for label_val in unique_labels:
            if label_val > 0:  # Skip background
                label_mask = labels == label_val
                color = self.get_label_color(label_val)
                overlay[label_mask] = color

        return overlay

    @staticmethod
    def strip_extensions(path: str) -> str:
        """Remove all suffixes from a file path."""
        p = Path(path)
        for _ in p.suffixes:
            p = p.with_suffix("")
        return str(p)

    def _normalize_output_path(self, out_path: str) -> tuple[str, str]:
        """Ensure path has a single valid extension and return (path, ext)."""
        p = Path(out_path)
        suffixes = p.suffixes
        if suffixes[-2:] == [".nii", ".gz"]:
            ext = ".nii.gz"
        elif suffixes:
            ext = suffixes[-1].lower()
        else:
            raise ValueError("Output path must have an extension")

        base = Path(self.strip_extensions(out_path))
        return str(base.with_suffix(ext)), ext

    def save_volume(self, data: np.ndarray, reference_path: str, out_path: str) -> str:
        """Save a 3D volume to disk using reference metadata."""
        out_path, ext = self._normalize_output_path(out_path)
        if ext in [".nii", ".nii.gz"]:
            ref_img = nib.load(reference_path)
            img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
            nib.save(img, out_path)
        elif ext in [".mha", ".mhd"]:
            if not SITK_AVAILABLE:
                raise RuntimeError(
                    "SimpleITK is required to save MHA/MHD files")
            ref_img = sitk.ReadImage(reference_path)
            # Transpose from nibabel's (X, Y, Z) to ITK's (Z, Y, X) convention
            arr = np.transpose(data, (2, 1, 0))
            itk_img = sitk.GetImageFromArray(arr)
            itk_img.SetSpacing(ref_img.GetSpacing())
            itk_img.SetDirection(ref_img.GetDirection())
            itk_img.SetOrigin(ref_img.GetOrigin())
            sitk.WriteImage(itk_img, out_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        return out_path

    def make_overlay_volume(self) -> np.ndarray:
        """Generate blended overlay volume from image and label data."""
        if self.image_data is None or self.label_data is None:
            return np.array([])

        img = self.image_data
        img_norm = (img - img.min()) / (img.ptp() if img.ptp() else 1) * 255
        img_norm = img_norm.astype(np.uint8)
        lbl = self.label_data.astype(np.int32)

        overlay_vol = np.zeros_like(img_norm, dtype=np.uint8)
        for k in range(img_norm.shape[2]):
            slice_img = img_norm[:, :, k]
            slice_lbl = lbl[:, :, k]
            color_ovr = self.create_label_overlay(slice_lbl)
            if color_ovr is not None:
                ovr_gray = (
                    0.299 * color_ovr[:, :, 0]
                    + 0.587 * color_ovr[:, :, 1]
                    + 0.114 * color_ovr[:, :, 2]
                ).astype(np.uint8)
                alpha = self._overlay_alpha
                blended = ((1 - alpha) * slice_img +
                           alpha * ovr_gray).astype(np.uint8)
            else:
                blended = slice_img
            overlay_vol[:, :, k] = blended

        return overlay_vol

    def render_slice(
        self,
        view_name: str,
        slice_idx: int,
        show_overlay: bool = True,
        alpha: float = 0.5,
    ) -> QImage:
        """Public rendering interface with LRU caching."""
        return self._render_slice_cached(
            view_name, slice_idx, show_overlay and self.global_overlay, alpha
        )

    @lru_cache(maxsize=100)
    def _render_slice_cached(
        self, view_name: str, slice_idx: int, show_overlay: bool, alpha: float
    ) -> QImage:
        """Internal slice renderer that is LRU cached."""
        config = self.view_configs[view_name]
        axis = config["axis"]

        img_slice = self.get_slice_data(axis, slice_idx, False)
        if img_slice.size == 0:
            return QImage()

        img_normalized = self.normalize_image(img_slice)

        # Handle overlay if available
        has_overlay = (
            show_overlay
            and self.label_data is not None
            and self.get_slice_data(axis, slice_idx, True).size > 0
        )

        if has_overlay:
            label_slice = self.get_slice_data(axis, slice_idx, True)
            overlay = self.create_label_overlay(label_slice)
            if overlay is not None:
                # Medical-grade alpha blending with ColorBrewer Paired colors
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
                mask = label_slice > 0

                # Use clinical-grade alpha with user adjustment
                # Base clinical alpha (0.4) modulated by user alpha setting
                clinical_alpha = (
                    self._overlay_alpha * alpha
                )  # Balanced clinical visibility
                img_rgb = img_rgb.astype(np.float32, copy=False)
                overlay_f = overlay.astype(np.float32, copy=False)

                # Alpha blend: base * (1-alpha) + overlay * alpha
                img_rgb[mask] = (1 - clinical_alpha) * img_rgb[
                    mask
                ] + clinical_alpha * overlay_f[mask]

                img_rgb = img_rgb.astype(np.uint8, copy=False)
                img_rgb = np.flipud(img_rgb)
                img_rgb = np.ascontiguousarray(img_rgb)
                height, width, _ = img_rgb.shape
                bytes_per_line = 3 * width
                qimage = QImage(
                    img_rgb.copy().data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888,
                )
            else:
                has_overlay = False

        if not has_overlay:
            # Use efficient grayscale format
            img_flipped = np.flipud(img_normalized)
            img_flipped = np.ascontiguousarray(img_flipped)
            height, width = img_flipped.shape
            qimage = QImage(
                img_flipped.copy().data,
                width,
                height,
                img_flipped.strides[0],
                QImage.Format_Grayscale8,
            )

        # Don't apply rotation to QImage - let pixmap handle rotation transform
        return qimage

    def load_image(self, filepath: str) -> None:
        """Load medical image data."""
        try:
            self.loadProgress.emit(0)
            self.image_data, self.image_affine, self.image_spacing = (
                self._load_image_data(filepath)
            )
            self.image_path = filepath
            self.loadProgress.emit(70)

            # Clear cache when new data is loaded
            self.get_slice_data.cache_clear()
            self._render_slice_cached.cache_clear()
            QPixmapCache.clear()

            # Reset slice positions
            for config in self.view_configs.values():
                config["slice"] = self.image_data.shape[config["axis"]] // 2

            self.loadProgress.emit(100)
            self.imageLoaded.emit(
                filepath, self.image_data.shape, self.image_affine, self.image_spacing
            )

        except Exception as e:
            self.loadError.emit(f"Failed to load image: {str(e)}")

    def load_labels(self, filepath: str) -> None:
        """Load medical label data."""
        try:
            self.loadProgress.emit(0)
            new_label_data, _, _ = self._load_image_data(filepath)
            new_label_data = new_label_data.astype(np.int32)
            self.loadProgress.emit(70)

            # Validate shape compatibility
            if (
                self.image_data is not None
                and new_label_data.shape != self.image_data.shape
            ):
                self.loadError.emit(
                    f"Label shape {new_label_data.shape} doesn't match "
                    f"image shape {self.image_data.shape}"
                )
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
        axis = self.view_configs[view_name]["axis"]
        return self.image_data.shape[axis] - 1

    def set_slice(self, view_name: str, slice_idx: int) -> None:
        """Update slice index for a view."""
        max_slice = self.get_max_slice(view_name)
        self.view_configs[view_name]["slice"] = max(
            0, min(slice_idx, max_slice))

    def get_label_at_position(self, view_name: str, x: int, y: int) -> int:
        """Get label value at specific pixel position."""
        if self.label_data is None:
            return 0

        config = self.view_configs[view_name]
        slice_idx = config["slice"]
        label_slice = self.get_slice_data(config["axis"], slice_idx, True)

        if label_slice.size == 0:
            return 0

        # Account for image flipping and bounds checking
        h, w = label_slice.shape
        y = h - 1 - y  # Flip Y coordinate

        if 0 <= x < w and 0 <= y < h:
            return int(label_slice[y, x])
        return 0

    def get_label_color(self, label_value: int) -> tuple[int, int, int]:
        """Get RGB color for a specific label value with medical imaging optimization."""
        if label_value == 0:
            return (0, 0, 0)  # Background is always black

        # Check if user has customized this label
        if label_value in self._custom_label_colors:
            base_color = self._custom_label_colors[label_value]
        else:
            # Use default ColorBrewer color cycling through the palette
            color_idx = (label_value - 1) % len(self._default_colors)
            base_color = self._default_colors[color_idx]

        # Apply medical imaging contrast enhancement
        return self._enhance_color_for_medical_display(base_color)

    def _enhance_color_for_medical_display(
        self, color: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """
        Enhance color contrast for medical imaging dark backgrounds.
        Boosts brightness and saturation while preserving hue relationships.
        """
        r, g, b = color

        # Convert to HSV for easier manipulation
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        diff = max_val - min_val

        # Calculate HSV values
        if diff == 0:
            hue = 0
        elif max_val == r_norm:
            hue = (60 * ((g_norm - b_norm) / diff) + 360) % 360
        elif max_val == g_norm:
            hue = (60 * ((b_norm - r_norm) / diff) + 120) % 360
        else:
            hue = (60 * ((r_norm - g_norm) / diff) + 240) % 360

        saturation = 0 if max_val == 0 else diff / max_val
        value = max_val

        # Medical imaging enhancement:
        # 1. Boost saturation for better distinction
        enhanced_saturation = min(1.0, saturation * 1.3)

        # 2. Increase brightness for dark background visibility
        enhanced_value = min(1.0, value * 1.2)

        # 3. Ensure minimum brightness for very dark colors
        enhanced_value = max(0.4, enhanced_value)

        # Convert back to RGB
        c = enhanced_value * enhanced_saturation
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = enhanced_value - c

        if 0 <= hue < 60:
            r_prime, g_prime, b_prime = c, x, 0
        elif 60 <= hue < 120:
            r_prime, g_prime, b_prime = x, c, 0
        elif 120 <= hue < 180:
            r_prime, g_prime, b_prime = 0, c, x
        elif 180 <= hue < 240:
            r_prime, g_prime, b_prime = 0, x, c
        elif 240 <= hue < 300:
            r_prime, g_prime, b_prime = x, 0, c
        else:
            r_prime, g_prime, b_prime = c, 0, x

        # Convert back to 0-255 range
        enhanced_r = int(min(255, max(0, (r_prime + m) * 255)))
        enhanced_g = int(min(255, max(0, (g_prime + m) * 255)))
        enhanced_b = int(min(255, max(0, (b_prime + m) * 255)))

        return (enhanced_r, enhanced_g, enhanced_b)

    def set_label_color(self, label_value: int, color: tuple[int, int, int]) -> None:
        """Set custom color for a specific label value."""
        if label_value > 0:  # Don't allow setting background color
            self._custom_label_colors[label_value] = color
            # Clear render cache to show new colors
            self._render_slice_cached.cache_clear()
            QPixmapCache.clear()

    def reset_label_colors(self) -> None:
        """Reset all custom colors to defaults."""
        self._custom_label_colors.clear()
        self._render_slice_cached.cache_clear()
        QPixmapCache.clear()

    def get_unique_label_values(self) -> list:
        """Get list of unique label values in current label data."""
        if self.label_data is None:
            return []
        unique_vals = np.unique(self.label_data).astype(int)
        # Filter out background (0) and sort
        return sorted([val for val in unique_vals if val > 0])

    def update_current_label_values(self) -> None:
        """Update the current label values list."""
        self._current_label_values = self.get_unique_label_values()


# ============================================================================
# MEASUREMENT LAYER - Measurement Management
# ============================================================================


class Measurement:
    """Data class for a single measurement."""

    def __init__(
        self,
        id: int,
        type: str,
        view_name: str,
        slice_idx: int,
        start_voxel: tuple[float, float, float],
        end_voxel: tuple[float, float, float],
        start_world: tuple[float, float, float],
        end_world: tuple[float, float, float],
        length_mm: float,
        timestamp: float,
        rotation: int = 0,
    ):
        self.id = id
        self.type = type  # 'line', 'angle', 'roi'
        self.view_name = view_name
        self.slice_idx = slice_idx
        self.start_voxel = start_voxel
        self.end_voxel = end_voxel
        self.start_world = start_world
        self.end_world = end_world
        self.length_mm = length_mm
        self.timestamp = timestamp
        self.rotation = rotation  # View rotation angle when measurement was taken

        # Style properties
        self.line_color = QColor(255, 0, 0)  # Default red
        self.line_width = 1.5  # Thinner default line width
        self.text_color = QColor(255, 0, 0)  # Default red
        self.text_font_size = 5.0  # Smaller default font size
        self.text_font_weight = QFont.Normal

        # Persistent layout fields (image coordinates)
        self.text_offset_img = None  # QPointF(dx, dy) relative to anchor
        self.text_anchor = "center"  # 'p1' | 'p2' | 'center'
        self.text_locked = False  # UI can toggle "lock label position"

    def apply_style(
        self,
        line_color=None,
        line_width=None,
        text_color=None,
        text_font_size=None,
        text_font_weight=None,
    ):
        """Apply style properties to this measurement."""
        if line_color is not None:
            self.line_color = line_color
        if line_width is not None:
            self.line_width = line_width
        if text_color is not None:
            self.text_color = text_color
        if text_font_size is not None:
            self.text_font_size = text_font_size
        if text_font_weight is not None:
            self.text_font_weight = text_font_weight


class MeasurementManager(QObject):
    """Manages all measurement data and related graphics items."""

    measurementAdded = Signal(object)  # measurement object
    measurementRemoved = Signal(int)  # measurement id
    settingsChanged = Signal()  # Emitted when measurement display settings change

    def __init__(self, model: ImageModel):
        super().__init__()
        self.model = model
        self.measurements: dict[int, Measurement] = {}
        self._next_id = 1

        # Graphics items tracking: measurement_id -> {view_name: [line_item, text_item]}
        self._graphics_items: dict[int, dict[str, list]] = {}

        # Joint nodes tracking: (view_name, slice_idx, key_x, key_y) -> Set[(measurement_id, handle_index)]
        # Enables endpoint snapping and linked movement
        self.joints: dict[tuple[str, int, int, int], set] = {}
        self.snap_distance = 12  # Pixel distance for joint clustering

        # Measurement display settings
        self.line_width = 1
        self.line_color = QColor(255, 0, 0)  # Red
        self.text_color = QColor(255, 0, 0)  # Red
        self.font_size = 6  # pt

    def add_line_measurement(
        self, view_name: str, slice_idx: int, start_pos: tuple, end_pos: tuple
    ):
        """Create and store a new line measurement."""
        if self.model.image_affine is None or self.model.image_spacing is None:
            return

        # Voxel coordinates
        start_voxel = self._get_voxel_coords(view_name, slice_idx, start_pos)
        end_voxel = self._get_voxel_coords(view_name, slice_idx, end_pos)

        # World coordinates
        start_world = nib.affines.apply_affine(
            self.model.image_affine, start_voxel)
        end_world = nib.affines.apply_affine(
            self.model.image_affine, end_voxel)

        # Length calculation
        length_mm = np.linalg.norm(np.array(start_world) - np.array(end_world))

        # Get current rotation angle for this view
        rotation = self.model.view_configs[view_name]["rotation"]

        measurement = Measurement(
            id=self._next_id,
            type="line",
            view_name=view_name,
            slice_idx=slice_idx,
            start_voxel=start_voxel,
            end_voxel=end_voxel,
            start_world=tuple(start_world),
            end_world=tuple(end_world),
            length_mm=length_mm,
            timestamp=time.time(),
            rotation=rotation,
        )

        self.measurements[self._next_id] = measurement
        self.measurementAdded.emit(measurement)

        # Register endpoints to joints for snapping and linked movement
        self._register_measurement_to_joints(
            self._next_id, view_name, slice_idx, start_pos, end_pos
        )

        self._next_id += 1

    def _get_voxel_coords(
        self, view_name: str, slice_idx: int, pos: tuple
    ) -> tuple[float, float, float]:
        """Convert 2D slice position to 3D voxel coordinates with sub-pixel precision."""
        x, y = pos
        axis = self.model.view_configs[view_name]["axis"]
        if axis == 0:  # Sagittal
            return (float(slice_idx), float(x), float(y))
        elif axis == 1:  # Coronal
            return (float(x), float(slice_idx), float(y))
        else:  # Axial
            return (float(x), float(y), float(slice_idx))

    def _get_joint_key(
        self, view_name: str, slice_idx: int, pos: tuple
    ) -> tuple[str, int, int, int]:
        """Convert image position to joint key for clustering nearby endpoints."""
        x, y = pos
        # 使用 floor 到像素，语义更稳定，避免边界处的四舍五入误并/误分
        key_x = int(x)
        key_y = int(y)
        return (view_name, slice_idx, key_x, key_y)

    def _register_measurement_to_joints(
        self,
        measurement_id: int,
        view_name: str,
        slice_idx: int,
        start_pos: tuple,
        end_pos: tuple,
    ):
        """Register measurement endpoints to joint system."""
        # Register start point (handle_index=0)
        start_key = self._get_joint_key(view_name, slice_idx, start_pos)
        if start_key not in self.joints:
            self.joints[start_key] = set()
        self.joints[start_key].add((measurement_id, 0))

        # Register end point (handle_index=1)
        end_key = self._get_joint_key(view_name, slice_idx, end_pos)
        if end_key not in self.joints:
            self.joints[end_key] = set()
        self.joints[end_key].add((measurement_id, 1))

    def _unregister_measurement_from_joints(self, measurement_id: int):
        """Remove measurement endpoints from joint system."""
        keys_to_remove = []
        for joint_key, endpoints in self.joints.items():
            # Remove all endpoints belonging to this measurement
            endpoints.discard((measurement_id, 0))
            endpoints.discard((measurement_id, 1))
            # Clean up empty joints
            if not endpoints:
                keys_to_remove.append(joint_key)

        for key in keys_to_remove:
            del self.joints[key]

    def _find_nearby_joint(
        self,
        view_name: str,
        slice_idx: int,
        pos: tuple,
        effective_snap_distance: float = None,
    ) -> tuple[str, int, int, int]:
        """Find existing joint within snap distance, or None."""
        x, y = pos
        snap_distance = (
            effective_snap_distance
            if effective_snap_distance is not None
            else self.snap_distance
        )
        for joint_key in self.joints.keys():
            if joint_key[0] != view_name or joint_key[1] != slice_idx:
                continue

            key_x, key_y = joint_key[2], joint_key[3]
            distance = ((x - key_x) ** 2 + (y - key_y) ** 2) ** 0.5
            if distance <= snap_distance:
                return joint_key
        return None

    def add_graphics_items(
        self,
        measurement_id: int,
        view_name: str,
        line_item,
        text_item,
        arrow_item=None,
        handle1=None,
        handle2=None,
    ):
        """Track graphics items for a measurement including draggable handles."""
        if measurement_id not in self._graphics_items:
            self._graphics_items[measurement_id] = {}
        if view_name not in self._graphics_items[measurement_id]:
            self._graphics_items[measurement_id][view_name] = []

        items = [line_item, text_item]
        if arrow_item:
            items.append(arrow_item)
        else:
            items.append(None)  # Keep consistent indexing
        if handle1:
            items.append(handle1)
        if handle2:
            items.append(handle2)
        self._graphics_items[measurement_id][view_name] = items

    def remove_graphics_items(self, measurement_id: int, view_name: str = None):
        """Remove graphics items for a measurement from specified view or all views."""
        if measurement_id not in self._graphics_items:
            return

        if view_name:
            # Remove from specific view
            if view_name in self._graphics_items[measurement_id]:
                items = self._graphics_items[measurement_id][view_name]
                for item in items:
                    if item and hasattr(item, "scene") and item.scene():
                        item.scene().removeItem(item)
                del self._graphics_items[measurement_id][view_name]
        else:
            # Remove from all views
            for view_items in self._graphics_items[measurement_id].values():
                for item in view_items:
                    if item and hasattr(item, "scene") and item.scene():
                        item.scene().removeItem(item)
            del self._graphics_items[measurement_id]

    def remove_measurement(self, measurement_id: int):
        """Remove a measurement by its ID."""
        if measurement_id in self.measurements:
            # Remove graphics first
            self.remove_graphics_items(measurement_id)

            # Unregister from joints system
            self._unregister_measurement_from_joints(measurement_id)

            # Remove data
            del self.measurements[measurement_id]
            self.measurementRemoved.emit(measurement_id)

    def refresh_measurement_display(self, controller):
        """Refresh measurement display in all views via controller."""
        for view_name in ["axial", "sagittal", "coronal"]:
            current_slice = self.model.view_configs[view_name]["slice"]
            self.update_measurement_display(
                view_name, current_slice, controller)

    def update_measurement_display(
        self, view_name: str, slice_idx: int, controller=None
    ):
        """Update measurement display for a specific view and slice."""
        # Remove all measurements from this view first
        for measurement_id in list(self._graphics_items.keys()):
            self.remove_graphics_items(measurement_id, view_name)

        # Add measurements that belong to this slice
        if controller:
            for measurement in self.measurements.values():
                if (
                    measurement.view_name == view_name
                    and measurement.slice_idx == slice_idx
                ):
                    controller._create_measurement_graphics(
                        measurement, view_name)

    def export_csv(self, filepath: str):
        """Export all measurements to a CSV file."""
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "ID",
                    "Type",
                    "View",
                    "Slice",
                    "Rotation (degrees)",
                    "Start Voxel X",
                    "Start Voxel Y",
                    "Start Voxel Z",
                    "End Voxel X",
                    "End Voxel Y",
                    "End Voxel Z",
                    "Start World X",
                    "Start World Y",
                    "Start World Z",
                    "End World X",
                    "End World Y",
                    "End World Z",
                    "Length (mm)",
                    "Timestamp",
                ]
            )
            for m in self.measurements.values():
                writer.writerow(
                    [
                        m.id,
                        m.type,
                        m.view_name,
                        m.slice_idx,
                        m.rotation,
                        m.start_voxel[0],
                        m.start_voxel[1],
                        m.start_voxel[2],
                        m.end_voxel[0],
                        m.end_voxel[1],
                        m.end_voxel[2],
                        m.start_world[0],
                        m.start_world[1],
                        m.start_world[2],
                        m.end_world[0],
                        m.end_world[1],
                        m.end_world[2],
                        m.length_mm,
                        m.timestamp,
                    ]
                )

    def set_line_width(self, width: int):
        if self.line_width != width:
            self.line_width = width
            self.settingsChanged.emit()

    def set_line_color(self, color: QColor):
        if self.line_color != color:
            self.line_color = color
            self.settingsChanged.emit()

    def set_text_color(self, color: QColor):
        if self.text_color != color:
            self.text_color = color
            self.settingsChanged.emit()

    def set_font_size(self, size: int):
        if self.font_size != size:
            self.font_size = size
            self.settingsChanged.emit()

    def apply_style_to_measurements(self, measurement_ids: list, **style_kwargs):
        """Apply style to specific measurements and refresh their display."""
        for measurement_id in measurement_ids:
            if measurement_id in self.measurements:
                self.measurements[measurement_id].apply_style(**style_kwargs)
        self.settingsChanged.emit()

    def apply_style_to_all_measurements(self, **style_kwargs):
        """Apply style to all measurements and refresh display."""
        for measurement in self.measurements.values():
            measurement.apply_style(**style_kwargs)
        self.settingsChanged.emit()


# ============================================================================
# DRAGGABLE GRAPHICS ITEMS - Interactive Measurement Components
# ============================================================================


class EndpointHandle(QGraphicsEllipseItem):
    """Draggable handle for measurement endpoints with position change callbacks."""

    def __init__(
        self,
        x,
        y,
        radius,
        measurement_id,
        handle_index,
        controller,
        view_name,
        parent=None,
    ):
        # 让句柄几何以原点为中心，位置用 setPos 控制
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, parent)
        self.measurement_id = measurement_id
        self.handle_index = handle_index  # 0 for start, 1 for end
        self.controller = controller
        self.view_name = view_name

        # 用位置放到目标点
        self.setPos(QPointF(x, y))

        # Visual styling
        self.setBrush(QBrush(QColor(255, 255, 0, 180)))
        self.setPen(QPen(QColor(0, 0, 0, 220), 1))
        self.setZValue(10)  # Above other items
        self.setData(0, measurement_id)

        # Enable dragging
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations,
                     True)  # 端点恒定屏幕大小
        self.setCursor(Qt.OpenHandCursor)

        # For Ctrl key detection to disable snapping
        self._disable_snapping = False

    def mousePressEvent(self, event):
        """Check for Ctrl key to disable snapping."""
        if event.button() == Qt.LeftButton:
            modifiers = QApplication.keyboardModifiers()
            self._disable_snapping = (modifiers & Qt.ControlModifier) != 0
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.controller:
            # 边界夹取要考虑半径（因为 pos() 表示圆心）
            if (
                hasattr(self.parentItem(), "pixmap")
                and not self.parentItem().pixmap().isNull()
            ):
                pixmap = self.parentItem().pixmap()
                rect = QRectF(0, 0, pixmap.width(), pixmap.height())
                r = self.rect().width() * 0.5  # 半径
                clamped = QPointF(
                    max(r, min(rect.width() - r, value.x())),
                    max(r, min(rect.height() - r, value.y())),
                )
                value = clamped

            # Call controller callback
            if hasattr(self.controller, "_on_handle_moved"):
                QTimer.singleShot(
                    0,
                    lambda: self.controller._on_handle_moved(
                        self.view_name,
                        self.measurement_id,
                        self.handle_index,
                        value,
                        self._disable_snapping,
                    ),
                )

        return super().itemChange(change, value)


class DraggableLine(QGraphicsLineItem):
    """Draggable line for moving entire measurements with position change callbacks."""

    def __init__(self, line, measurement_id, controller, view_name, parent=None):
        super().__init__(line, parent)
        self.measurement_id = measurement_id
        self.controller = controller
        self.view_name = view_name

        # Enable dragging
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setCursor(Qt.SizeAllCursor)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.controller:
            # 以"尝试位移量"作为 delta，驱动数据更新
            delta = value - self.pos()
            if delta.manhattanLength() != 0:
                QTimer.singleShot(
                    0,
                    lambda: self.controller._on_line_moved(
                        self.view_name, self.measurement_id, delta
                    ),
                )
            # 关键：阻止 item 累积位移，返回当前位置，不让它真的移动
            return self.pos()
        return super().itemChange(change, value)


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

    def run(self) -> None:
        """Execute loading in background thread."""
        try:
            if self.is_label:
                self.model.load_labels(self.filepath)
            else:
                self.model.load_image(self.filepath)
        except Exception as e:
            self.model.loadError.emit(f"Background loading failed: {str(e)}")


class PreloadWorker(QRunnable):
    """Background worker for preloading adjacent slices."""

    def __init__(
        self,
        model: ImageModel,
        view_name: str,
        slice_indices: list,
        show_overlay: bool = True,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.view_name = view_name
        self.slice_indices = slice_indices
        self.show_overlay = show_overlay
        self.alpha = alpha

    def run(self) -> None:
        """Preload slices in background."""
        try:
            for slice_idx in self.slice_indices:
                # Check if slice is valid
                max_slice = self.model.get_max_slice(self.view_name)
                if 0 <= slice_idx <= max_slice:
                    # Trigger cache population via render_slice
                    self.model.render_slice(
                        self.view_name, slice_idx, self.show_overlay, self.alpha
                    )
        except Exception as e:
            # Silent fail for preloading to avoid disrupting user experience
            pass


# ============================================================================
# VIEW LAYER - UI Components
# ============================================================================


class AboutDialog(QDialog):
    """
    About dialog with trilingual (English/Chinese/French) support and software information.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("about_title"))
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
        self.create_french_tab()

        # Close button
        close_btn = QPushButton(tr("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # Apply dialog styles
        self.setStyleSheet(
            """
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
        """
        )

    def create_english_tab(self) -> None:
        """Create English content tab."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Main title
        title_label = QLabel("Medical Image Viewer")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #4a90e2; margin-bottom: 10px;"
        )
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Version info
        version_label = QLabel("Version 0.1.4")
        version_label.setStyleSheet(
            "font-size: 14px; color: #ccc; margin-bottom: 15px;"
        )
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # Description
        description = QTextEdit()
        description.setReadOnly(True)
        description.setMaximumHeight(300)
        description.setHtml(
            """
        <h3 style="color: #4a90e2;">What is this Viewer?</h3>
        <p>A simple, fast viewer for medical images. View MRI scans, overlays, and segmentation masks 
        in three perspectives simultaneously.</p>
        
        <h3 style="color: #4a90e2;">How to Use</h3>
        <ul>
            <li><b>Load Files:</b> Use File menu → Load Image/Labels, or type paths in the right panel</li>
            <li><b>Navigate:</b> Mouse wheel scrolls through slices, Ctrl+wheel zooms in/out</li>
            <li><b>Pan & Rotate:</b> Right-click drag to move view, click rotation buttons to flip</li>
            <li><b>Overlays:</b> Check "Show Overlay" and adjust transparency slider</li>
            <li><b>Save:</b> File → Save Screenshot (Ctrl+S) or Volume (Ctrl+Shift+S)</li>
        </ul>
        
        <h3 style="color: #4a90e2;">Keyboard Shortcuts</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="padding: 4px;"><b>Ctrl+O</b></td><td style="padding: 4px;">Open image file</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+L</b></td><td style="padding: 4px;">Open label file</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+S</b></td><td style="padding: 4px;">Save screenshot</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+Shift+S</b></td><td style="padding: 4px;">Open volume save menu</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+R</b></td><td style="padding: 4px;">Reset views and clear cache</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+T</b></td><td style="padding: 4px;">Toggle control panel</td></tr>
            <tr><td style="padding: 4px;"><b>F</b></td><td style="padding: 4px;">Fit all views to window</td></tr>
            <tr><td style="padding: 4px;"><b>F1</b></td><td style="padding: 4px;">Show this dialog</td></tr>
            <tr><td style="padding: 4px;"><b>Wheel</b></td><td style="padding: 4px;">Scroll through slices</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+Wheel</b></td><td style="padding: 4px;">Zoom in/out</td></tr>
            <tr><td style="padding: 4px;"><b>Right-drag</b></td><td style="padding: 4px;">Pan view</td></tr>
        </table>
        
        <h3 style="color: #4a90e2;">Command Line Options</h3>
        <p>Run <code>python nifti_viewer.py --help</code> for full options.<br>
        Examples: <code>-i image.mha -l labels.nii.gz</code></p>
        
        <h3 style="color: #4a90e2;">File Support</h3>
        <p>Supports NIfTI (.nii, .nii.gz) and MetaImage (.mha, .mhd) formats.</p>
        """
        )
        layout.addWidget(description)

        scroll_area.setWidget(content_widget)
        self.tab_widget.addTab(scroll_area, tr("english"))

    def create_chinese_tab(self) -> None:
        """Create Chinese content tab."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Main title
        title_label = QLabel("医学影像查看器")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #4a90e2; margin-bottom: 10px;"
        )
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Version info
        version_label = QLabel("版本 0.1.4")
        version_label.setStyleSheet(
            "font-size: 14px; color: #ccc; margin-bottom: 15px;"
        )
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # Description
        description = QTextEdit()
        description.setReadOnly(True)
        description.setMaximumHeight(300)
        description.setHtml(
            """
        <h3 style="color: #4a90e2;">什么是医学影像查看器？</h3>
        <p>简单快速的医学影像查看工具。支持查看磁共振(MRI)扫描图像、叠加图层和分割掩膜，
        同时显示三个角度的切面视图。</p>
        
        <h3 style="color: #4a90e2;">如何使用</h3>
        <ul>
            <li><b>加载文件：</b>使用文件菜单 → 加载影像/标签，或在右侧面板输入文件路径</li>
            <li><b>导航操作：</b>鼠标滚轮切换切片，Ctrl+滚轮缩放视图</li>
            <li><b>平移旋转：</b>右键拖拽移动视图，点击旋转按钮翻转方向</li>
            <li><b>叠加显示：</b>勾选"显示叠加"并调节透明度滑条</li>
            <li><b>保存：</b>文件 → 保存截图 (Ctrl+S) 或保存体数据 (Ctrl+Shift+S)</li>
        </ul>
        
        <h3 style="color: #4a90e2;">快捷键一览</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="padding: 4px;"><b>Ctrl+O</b></td><td style="padding: 4px;">打开影像文件</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+L</b></td><td style="padding: 4px;">打开标签文件</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+S</b></td><td style="padding: 4px;">保存截图</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+Shift+S</b></td><td style="padding: 4px;">打开体数据保存菜单</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+R</b></td><td style="padding: 4px;">重置视图并清空缓存</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+T</b></td><td style="padding: 4px;">切换控制面板显示</td></tr>
            <tr><td style="padding: 4px;"><b>F</b></td><td style="padding: 4px;">适应所有视图到窗口</td></tr>
            <tr><td style="padding: 4px;"><b>F1</b></td><td style="padding: 4px;">显示此对话框</td></tr>
            <tr><td style="padding: 4px;"><b>滚轮</b></td><td style="padding: 4px;">切换切片</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+滚轮</b></td><td style="padding: 4px;">缩放视图</td></tr>
            <tr><td style="padding: 4px;"><b>右键拖拽</b></td><td style="padding: 4px;">平移视图</td></tr>
        </table>
        
        <h3 style="color: #4a90e2;">命令行选项</h3>
        <p>运行 <code>python nifti_viewer.py --help</code> 查看完整选项。<br>
        示例：<code>-i image.mha -l labels.nii.gz</code></p>
        
        <h3 style="color: #4a90e2;">文件支持</h3>
        <p>支持 NIfTI (.nii, .nii.gz) 和 MetaImage (.mha, .mhd) 格式。</p>
        """
        )
        layout.addWidget(description)

        scroll_area.setWidget(content_widget)
        self.tab_widget.addTab(scroll_area, tr("chinese"))

    def create_french_tab(self) -> None:
        """Create French content tab."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Main title
        title_label = QLabel("Visionneuse d'Images Médicales")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #4a90e2; margin-bottom: 10px;"
        )
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Version info
        version_label = QLabel("Version 0.1.4")
        version_label.setStyleSheet(
            "font-size: 14px; color: #ccc; margin-bottom: 15px;"
        )
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # Description
        description = QTextEdit()
        description.setReadOnly(True)
        description.setMaximumHeight(300)
        description.setHtml(
            """
        <h3 style="color: #4a90e2;">Qu'est-ce que cette Visionneuse ?</h3>
        <p>Une visionneuse simple et rapide pour les images médicales. Visualisez les examens IRM, 
        les superpositions et les masques de segmentation dans trois perspectives simultanément.</p>
        
        <h3 style="color: #4a90e2;">Comment l'Utiliser</h3>
        <ul>
            <li><b>Charger des Fichiers :</b> Utilisez le menu Fichier → Charger Image/Étiquettes, ou tapez les chemins dans le panneau de droite</li>
            <li><b>Naviguer :</b> La molette de la souris fait défiler les coupes, Ctrl+molette zoome</li>
            <li><b>Panoramique et Rotation :</b> Clic droit et glisser pour déplacer la vue, cliquez sur les boutons de rotation pour retourner</li>
            <li><b>Superpositions :</b> Cochez "Afficher la Superposition" et ajustez le curseur de transparence</li>
            <li><b>Enregistrer :</b> Fichier → Enregistrer Capture d'Écran (Ctrl+S) ou Volume (Ctrl+Shift+S)</li>
        </ul>
        
        <h3 style="color: #4a90e2;">Raccourcis Clavier</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="padding: 4px;"><b>Ctrl+O</b></td><td style="padding: 4px;">Ouvrir un fichier image</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+L</b></td><td style="padding: 4px;">Ouvrir un fichier d'étiquettes</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+S</b></td><td style="padding: 4px;">Enregistrer capture d'écran</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+Shift+S</b></td><td style="padding: 4px;">Ouvrir le menu d'enregistrement de volume</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+R</b></td><td style="padding: 4px;">Réinitialiser les vues et vider le cache</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+T</b></td><td style="padding: 4px;">Basculer le panneau de contrôle</td></tr>
            <tr><td style="padding: 4px;"><b>F</b></td><td style="padding: 4px;">Ajuster toutes les vues à la fenêtre</td></tr>
            <tr><td style="padding: 4px;"><b>F1</b></td><td style="padding: 4px;">Afficher cette boîte de dialogue</td></tr>
            <tr><td style="padding: 4px;"><b>Molette</b></td><td style="padding: 4px;">Faire défiler les coupes</td></tr>
            <tr><td style="padding: 4px;"><b>Ctrl+Molette</b></td><td style="padding: 4px;">Zoomer/Dézoomer</td></tr>
            <tr><td style="padding: 4px;"><b>Clic droit-glisser</b></td><td style="padding: 4px;">Panoramique de la vue</td></tr>
        </table>
        
        <h3 style="color: #4a90e2;">Options de Ligne de Commande</h3>
        <p>Exécutez <code>python nifti_viewer.py --help</code> pour les options complètes.<br>
        Exemples : <code>-i image.mha -l labels.nii.gz</code></p>
        
        <h3 style="color: #4a90e2;">Support de Fichiers</h3>
        <p>Supporte les formats NIfTI (.nii, .nii.gz) et MetaImage (.mha, .mhd).</p>
        """
        )
        layout.addWidget(description)

        scroll_area.setWidget(content_widget)
        self.tab_widget.addTab(scroll_area, tr("french"))


class SliceView(QGraphicsView):
    """
    Enhanced QGraphicsView for medical image slice display.
    Supports zooming, panning, rotation, and label tooltips.
    """

    # Signals
    mousePositionChanged = Signal(int, int)  # x, y coordinates
    wheelScrolled = Signal(int)  # delta for slice navigation
    # start_pos, end_pos in image coordinates
    lineMeasured = Signal(object, object)

    def __init__(self, view_name: str, parent=None):
        super().__init__(parent)

        self.view_name = view_name
        self.model: ImageModel | None = None
        self.measure_mode = "off"  # 'off', 'line', 'erase'
        self._measure_points = []
        self._measure_temp_item = None

        # Snap system properties
        self.snap_enabled = True
        self.snap_distance = 12  # pixels for screen-space constant snapping
        self._snap_preview_item = None
        self._current_snap_point = None

        # Hover highlight system
        self._hover_highlight_enabled = True
        self._currently_highlighted_measurement_id = None

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

    def set_model(self, model: ImageModel) -> None:
        """Connect to data model."""
        self.model = model

    def set_image(self, qimage: QImage, rotation: int = 0) -> None:
        """Update displayed image with QPixmapCache optimization and handle rotation via transform."""
        if qimage.isNull():
            self.pixmap_item.setPixmap(QPixmap())
            self.pixmap_item.setTransform(QTransform())  # Reset transform
            return

        # Generate cache key based on image properties and view name (without rotation)
        cache_key = f"{self.view_name}_{qimage.cacheKey()}_{qimage.format()}"
        pixmap = QPixmapCache.find(cache_key)

        if pixmap is None:
            pixmap = QPixmap.fromImage(qimage)
            # Insert with size-aware caching
            QPixmapCache.insert(cache_key, pixmap)

        self.pixmap_item.setPixmap(pixmap)

        # Apply rotation as pixmap transform instead of image transform
        if rotation != 0:
            transform = QTransform()
            transform.rotate(rotation)
            self.pixmap_item.setTransform(transform)
        else:
            self.pixmap_item.setTransform(QTransform())  # Reset to identity

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

            # Force recalculate measurements after scaling
            # Text uses ItemIgnoresTransformations (constant screen size),
            # but arrows need to be recalculated for the new scaling
            if hasattr(self, "controller") and self.controller:
                if hasattr(self.controller, "model") and self.controller.model:
                    slice_idx = self.controller.model.view_configs[self.view_name][
                        "slice"
                    ]
                    self.controller.measurement_manager.update_measurement_display(
                        self.view_name, slice_idx, self.controller
                    )
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
        elif event.button() == Qt.LeftButton and self.measure_mode == "line":
            # Use snap point if available, otherwise use mouse position
            if self._current_snap_point:
                start_img_pos = self._current_snap_point
            else:
                start_img_pos = self.map_view_to_image_coords(event.position())
            self._measure_points = [start_img_pos]

            # Flip Y for drawing on scene (which is already flipped)
            h = self.pixmap_item.pixmap().height()
            start_draw_y = h - 1 - start_img_pos.y()

            line = QGraphicsLineItem(
                start_img_pos.x(),
                start_draw_y,
                start_img_pos.x(),
                start_draw_y,
                parent=self.pixmap_item,
            )
            temp_pen = QPen(QColor(255, 255, 0), 2)
            temp_pen.setCosmetic(True)  # 临时线条也恒定粗细
            line.setPen(temp_pen)
            self._measure_temp_item = line
        elif event.button() == Qt.LeftButton and self.measure_mode == "erase":
            # Delete measurement under cursor
            self._delete_measurement_under_cursor(event.position())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement for panning, tooltips, and eraser highlighting."""
        if self._panning:
            # Pan the view
            delta = event.position() - self._pan_start
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            self._pan_start = event.position()
        elif self.measure_mode == "erase":
            # Highlight measurement items under cursor
            self._highlight_measurements_under_cursor(event.position())
        elif self.measure_mode == "line":
            # Check for snap points when in line mode
            snap_point = self._find_snap_point(event.position())
            if snap_point:
                self._show_snap_preview(snap_point)
                self._current_snap_point = snap_point
            else:
                self._clear_snap_preview()
                self._current_snap_point = None

            # Update temporary line if drawing
            if self._measure_temp_item:
                p2_img = self.map_view_to_image_coords(event.position())

                # Flip Y for drawing on scene
                h = self.pixmap_item.pixmap().height()
                p2_draw_y = h - 1 - p2_img.y()

                self._measure_temp_item.setLine(
                    self._measure_points[0].x(),
                    h - 1 - self._measure_points[0].y(),
                    p2_img.x(),
                    p2_draw_y,
                )
        else:
            # Emit position for tooltip
            scene_pos = self.mapToScene(event.position().toPoint())
            item_pos = self.pixmap_item.mapFromScene(scene_pos)

            if self.pixmap_item.contains(item_pos):
                x, y = int(item_pos.x()), int(item_pos.y())
                self.mousePositionChanged.emit(x, y)

            # Handle measurement hover highlighting in normal mode
            if self._hover_highlight_enabled:
                self._handle_measurement_hover(event.position())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        elif (
            event.button() == Qt.LeftButton
            and self.measure_mode == "line"
            and self._measure_temp_item
        ):
            # Use snap point if available, otherwise use mouse position
            if self._current_snap_point:
                end_img_pos = self._current_snap_point
            else:
                end_img_pos = self.map_view_to_image_coords(event.position())
            self._measure_points.append(end_img_pos)
            self.scene.removeItem(self._measure_temp_item)
            self._measure_temp_item = None
            # Clear snap preview
            self._clear_snap_preview()
            self._current_snap_point = None
            self.lineMeasured.emit(
                self._measure_points[0], self._measure_points[1])
            self._measure_points = []

        super().mouseReleaseEvent(event)

    def map_view_to_image_coords(self, view_pos: QPointF) -> QPointF:
        """Map a point from view coordinates to original image coordinates."""
        # Map from view to scene
        scene_pos = self.mapToScene(view_pos.toPoint())

        # Map from scene to the pixmap item (which is the displayed slice)
        item_pos = self.pixmap_item.mapFromScene(scene_pos)

        # The pixmap might be transformed (rotated). We need the inverse transform.
        img_to_item_transform = self.pixmap_item.transform()
        item_to_img_transform, _ = img_to_item_transform.inverted()
        img_pos = item_to_img_transform.map(item_pos)

        # The image was flipped vertically during rendering. We need to un-flip it.
        h = self.pixmap_item.pixmap().height()
        final_pos = QPointF(img_pos.x(), h - 1 - img_pos.y())

        return final_pos

    def _find_snap_point(self, mouse_pos: QPointF) -> QPointF | None:
        """Find the nearest snap point within snap distance."""
        if (
            not self.snap_enabled
            or not hasattr(self, "controller")
            or not self.controller
        ):
            return None

        # Use item coordinates for consistent comparison
        mouse_img = self.map_view_to_image_coords(mouse_pos)
        nearest_point = None

        # Calculate screen-space constant snap distance
        scale = self.transform().m11()
        effective_snap_distance = self.snap_distance / max(scale, 1e-6)
        min_distance = effective_snap_distance

        if (
            not hasattr(self.pixmap_item, "pixmap")
            or self.pixmap_item.pixmap().isNull()
        ):
            return None

        h = self.pixmap_item.pixmap().height()
        mouse_item = QPointF(mouse_img.x(), h - 1 - mouse_img.y())

        # Check all measurement endpoints
        for measurement in self.controller.measurement_manager.measurements.values():
            if measurement.view_name != self.view_name:
                continue

            # Only check measurements on current slice
            current_slice = (
                self.model.view_configs[self.view_name]["slice"] if self.model else 0
            )
            if measurement.slice_idx != current_slice:
                continue

            # Get measurement endpoints in image coordinates
            axis = self.model.view_configs[self.view_name]["axis"] if self.model else 2
            if axis == 0:  # Sagittal
                p1_img = QPointF(
                    measurement.start_voxel[1], measurement.start_voxel[2])
                p2_img = QPointF(
                    measurement.end_voxel[1], measurement.end_voxel[2])
            elif axis == 1:  # Coronal
                p1_img = QPointF(
                    measurement.start_voxel[0], measurement.start_voxel[2])
                p2_img = QPointF(
                    measurement.end_voxel[0], measurement.end_voxel[2])
            else:  # Axial
                p1_img = QPointF(
                    measurement.start_voxel[0], measurement.start_voxel[1])
                p2_img = QPointF(
                    measurement.end_voxel[0], measurement.end_voxel[1])

            # Convert to item coordinates (with Y-flip) for distance calculation
            p1_item = QPointF(p1_img.x(), h - 1 - p1_img.y())
            p2_item = QPointF(p2_img.x(), h - 1 - p2_img.y())

            # Calculate distances in item coordinates
            dist1 = (
                (mouse_item.x() - p1_item.x()) ** 2
                + (mouse_item.y() - p1_item.y()) ** 2
            ) ** 0.5
            dist2 = (
                (mouse_item.x() - p2_item.x()) ** 2
                + (mouse_item.y() - p2_item.y()) ** 2
            ) ** 0.5

            if dist1 < min_distance:
                min_distance = dist1
                nearest_point = p1_img  # Return in image coordinates
            if dist2 < min_distance:
                min_distance = dist2
                nearest_point = p2_img  # Return in image coordinates

        return nearest_point

    def _show_snap_preview(self, snap_point: QPointF):
        """Show visual preview of snap point."""
        self._clear_snap_preview()

        if (
            hasattr(self.pixmap_item, "pixmap")
            and not self.pixmap_item.pixmap().isNull()
        ):
            # Convert to item coordinates for drawing
            h = self.pixmap_item.pixmap().height()
            item_point = QPointF(snap_point.x(), h - 1 - snap_point.y())

            # Create snap preview circle (radius matches handle radius) as child of pixmap_item
            radius = 5  # Match handle radius
            # 以原点为中心创建，再用setPos定位，配合ItemIgnoresTransformations
            self._snap_preview_item = QGraphicsEllipseItem(
                -radius, -radius, radius * 2, radius * 2, parent=self.pixmap_item
            )
            self._snap_preview_item.setPos(item_point)
            self._snap_preview_item.setFlag(
                QGraphicsItem.ItemIgnoresTransformations, True
            )  # 恒定屏幕大小
            pen = QPen(QColor(255, 255, 0, 200), 2)
            pen.setCosmetic(True)  # 描边也恒定粗细
            self._snap_preview_item.setPen(pen)
            self._snap_preview_item.setBrush(
                QBrush(QColor(255, 255, 0, 100))
            )  # Semi-transparent fill

    def _clear_snap_preview(self):
        """Clear snap preview graphics unconditionally."""
        if self._snap_preview_item:
            # Unconditionally remove from scene if present - fixed version
            if self._snap_preview_item.scene():
                self._snap_preview_item.scene().removeItem(self._snap_preview_item)
            self._snap_preview_item = None

    def _highlight_measurements_under_cursor(self, view_pos: QPointF):
        """Highlight measurement graphics under cursor for eraser tool."""
        # Clear previous highlights
        for item in self.scene.items():
            if hasattr(item, "setOpacity") and item.data(0):  # It's a measurement item
                item.setOpacity(1.0)  # Reset to normal

        # Find items under cursor
        scene_pos = self.mapToScene(view_pos.toPoint())
        items_under_cursor = self.scene.items(
            scene_pos, Qt.IntersectsItemShape, Qt.DescendingOrder
        )

        # Highlight measurement items
        for item in items_under_cursor:
            if item.data(0):  # It's a measurement item with measurement ID
                item.setOpacity(0.5)  # Highlight by reducing opacity
                self.setCursor(Qt.PointingHandCursor)
                return

        # No measurement under cursor
        self.setCursor(Qt.ArrowCursor)

    def _delete_measurement_under_cursor(self, view_pos: QPointF):
        """Delete measurement graphics under cursor."""
        scene_pos = self.mapToScene(view_pos.toPoint())
        items_under_cursor = self.scene.items(
            scene_pos, Qt.IntersectsItemShape, Qt.DescendingOrder
        )

        # Find measurement item and delete
        for item in items_under_cursor:
            measurement_id = item.data(0)
            if measurement_id:  # It's a measurement item
                # Emit signal to controller to delete the measurement
                if hasattr(self, "controller") and self.controller:
                    self.controller.measurement_manager.remove_measurement(
                        measurement_id
                    )
                break

    def leaveEvent(self, event):
        """Clear highlights when mouse leaves the view."""
        if self.measure_mode == "erase":
            # Clear all highlights and reset cursor
            for item in self.scene.items():
                if hasattr(item, "setOpacity") and item.data(
                    0
                ):  # It's a measurement item
                    item.setOpacity(1.0)  # Reset to normal
            self.setCursor(Qt.ArrowCursor)
        elif self.measure_mode == "line":
            # Clear snap preview when mouse leaves
            self._clear_snap_preview()
            self._current_snap_point = None
        else:
            # Clear measurement hover highlights in normal mode
            if self._hover_highlight_enabled:
                self._clear_measurement_highlights()
        super().leaveEvent(event)

    def _handle_measurement_hover(self, view_pos: QPointF):
        """Handle measurement hover highlighting in normal mode."""
        # Get item under cursor
        scene_pos = self.mapToScene(view_pos.toPoint())
        items_under_cursor = self.scene.items(scene_pos)

        measurement_id = None
        # Find measurement item under cursor
        for item in items_under_cursor:
            if item.data(0) and isinstance(
                item, (QGraphicsLineItem,
                       QGraphicsSimpleTextItem, QGraphicsPathItem)
            ):
                measurement_id = item.data(0)
                break

        # Update highlight if measurement changed
        if measurement_id != self._currently_highlighted_measurement_id:
            self._clear_measurement_highlights()
            if measurement_id:
                self._highlight_measurement_group(measurement_id)
            self._currently_highlighted_measurement_id = measurement_id

    def _highlight_measurement_group(self, measurement_id: int):
        """Highlight all graphics items belonging to a measurement (line + text + arrow)."""
        if not hasattr(self, "controller") or not self.controller:
            return

        # Get all items for this measurement
        measurement_items = []
        for item in self.scene.items():
            if item.data(0) == measurement_id and isinstance(
                item, (QGraphicsLineItem,
                       QGraphicsSimpleTextItem, QGraphicsPathItem)
            ):
                measurement_items.append(item)

        # Apply medical-grade highlight (outline effect, not transparency)
        # Bright yellow for high contrast
        highlight_color = QColor(255, 255, 0)
        highlight_width = 2.0  # Bold outline

        for item in measurement_items:
            if isinstance(item, QGraphicsLineItem):
                # Highlight line with bright outline
                current_pen = item.pen()
                highlight_pen = QPen(
                    highlight_color, current_pen.widthF() + highlight_width
                )
                highlight_pen.setCosmetic(True)  # 高亮线条也恒定粗细
                item.setPen(highlight_pen)
            elif isinstance(item, QGraphicsSimpleTextItem):
                # Highlight text with bright outline
                text_pen = QPen(highlight_color, highlight_width)
                text_pen.setCosmetic(True)  # 文字描边也恒定粗细
                item.setPen(text_pen)
            elif isinstance(item, QGraphicsPathItem):
                # Highlight arrow with bright outline
                current_pen = item.pen()
                highlight_pen = QPen(
                    highlight_color, current_pen.widthF() + 1.0)
                highlight_pen.setCosmetic(True)  # 高亮箭头也恒定粗细
                item.setPen(highlight_pen)

    def _clear_measurement_highlights(self):
        """Clear all measurement highlights and restore original styles."""
        if not hasattr(self, "controller") or not self.controller:
            return

        # Clear highlights for currently highlighted measurement
        if self._currently_highlighted_measurement_id:
            measurement = self.controller.measurement_manager.measurements.get(
                self._currently_highlighted_measurement_id
            )
            if measurement:
                # Restore original styles
                for item in self.scene.items():
                    if item.data(0) == self._currently_highlighted_measurement_id:
                        if isinstance(item, QGraphicsLineItem):
                            # Restore original line style
                            original_pen = QPen(
                                measurement.line_color, measurement.line_width
                            )
                            original_pen.setCosmetic(True)  # 恢复时也保持恒定粗细
                            item.setPen(original_pen)
                        elif isinstance(item, QGraphicsSimpleTextItem):
                            # Clear text outline
                            item.setPen(QPen(Qt.NoPen))
                        elif isinstance(item, QGraphicsPathItem):
                            # Restore elegant arrow style using measurement's text color
                            original_pen = QPen(
                                measurement.text_color,
                                max(1.0, measurement.line_width * 0.9),
                            )
                            original_pen.setCosmetic(True)  # 恢复时也保持恒定粗细
                            item.setPen(original_pen)

        self._currently_highlighted_measurement_id = None

    def reset_view(self) -> None:
        """Reset zoom and pan to fit image."""
        self.resetTransform()
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)


class MainWindow(QMainWindow):
    """
    Main application window with menu, three-view layout, and controls.
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.setWindowTitle(tr("window_title"))
        self.setWindowIcon(QIcon("media/logo.png"))
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)  # Set a good default size

        # Create components
        self.setup_ui()
        self.setup_toolbar()
        self.setup_menu()
        self.setup_statusbar()
    
    def update_ui_text(self):
        """Update all UI text with current language translations."""
        self.setWindowTitle(tr("window_title"))
        
        # Update menu text
        if hasattr(self, 'menuBar'):
            for action in self.menuBar().actions():
                menu = action.menu()
                if menu:
                    if menu.title() in ["File", "文件", "Fichier"]:
                        menu.setTitle(tr("menu_file"))
                    elif menu.title() in ["View", "视图", "Affichage"]:
                        menu.setTitle(tr("menu_view"))
                    elif menu.title() in ["Help", "帮助", "Aide"]:
                        menu.setTitle(tr("menu_help"))
                    elif menu.title() in ["Language", "语言", "Langue"]:
                        menu.setTitle(tr("language"))
        
        # Update actions
        if hasattr(self, 'actions'):
            self.actions["load_image"].setText(tr("load_image"))
            self.actions["load_labels"].setText(tr("load_labels"))
            self.actions["save_image"].setText(tr("save_image"))
            self.actions["save_label"].setText(tr("save_label"))
            self.actions["save_overlay"].setText(tr("save_overlay"))
            self.actions["save_screenshot"].setText(tr("save_screenshot"))
            self.actions["reset"].setText(tr("reset"))
            self.actions["exit"].setText(tr("exit"))
            self.actions["fit_all"].setText(tr("fit_all_views"))
            self.actions["toggle_control_panel"].setText(tr("toggle_control_panel"))
            self.actions["about"].setText(tr("about"))
        
        # Update control dock
        if hasattr(self, 'control_dock'):
            self.control_dock.setWindowTitle(tr("controls"))
        
        # Update control buttons
        if hasattr(self, 'load_image_btn'):
            self.load_image_btn.setText(tr("load_image_btn"))
        if hasattr(self, 'load_labels_btn'):
            self.load_labels_btn.setText(tr("load_labels_btn"))
        if hasattr(self, 'reset_btn'):
            self.reset_btn.setText(tr("reset_btn"))
        if hasattr(self, 'update_image_btn'):
            self.update_image_btn.setText(tr("update_image"))
        if hasattr(self, 'update_labels_btn'):
            self.update_labels_btn.setText(tr("update_labels"))
        
        # Update status bar
        if hasattr(self, 'status_label'):
            current_text = self.status_label.text()
            if "Ready" in current_text or "就绪" in current_text or "Prêt" in current_text:
                self.status_label.setText(tr("status_ready"))
        
        # Update view titles
        view_titles = [tr("axial_view"), tr("sagittal_view"), tr("coronal_view")]
        view_names = ["axial", "sagittal", "coronal"]
        
        if hasattr(self, 'slice_views'):
            for i, name in enumerate(view_names):
                # Find the title label for this view
                view_widget = self.slice_views[name].parent()
                if view_widget:
                    layout = view_widget.layout()
                    if layout and layout.count() > 0:
                        title_widget = layout.itemAt(0).widget()
                        if isinstance(title_widget, QLabel):
                            title_widget.setText(view_titles[i])
        
        # Update control panel elements
        if hasattr(self, 'image_path_input'):
            self.image_path_input.setPlaceholderText(tr("no_image_loaded"))
        if hasattr(self, 'label_path_input'):
            self.label_path_input.setPlaceholderText(tr("no_labels_loaded_placeholder"))
        if hasattr(self, 'image_path_label'):
            self.image_path_label.setText(tr("image_path"))
        if hasattr(self, 'label_path_label'):
            self.label_path_label.setText(tr("label_path"))
            
        # Update checkboxes
        if hasattr(self, 'view_checkboxes'):
            for name, checkbox in self.view_checkboxes.items():
                checkbox.setText(tr(name))
        if hasattr(self, 'global_overlay_cb'):
            self.global_overlay_cb.setText(tr("show_overlay"))
        if hasattr(self, 'snap_enabled_checkbox'):
            self.snap_enabled_checkbox.setText(tr("enable_endpoint_snapping"))
            
        # Update buttons
        if hasattr(self, 'reset_colors_btn'):
            self.reset_colors_btn.setText(tr("reset_to_defaults"))
        if hasattr(self, 'line_tool_btn'):
            self.line_tool_btn.setText(tr("line_tool"))
        if hasattr(self, 'eraser_tool_btn'):
            self.eraser_tool_btn.setText(tr("eraser"))
        if hasattr(self, 'export_measurements_btn'):
            self.export_measurements_btn.setText(tr("export_csv"))
        if hasattr(self, 'select_all_btn'):
            self.select_all_btn.setText(tr("select_all"))
        if hasattr(self, 'deselect_all_btn'):
            self.deselect_all_btn.setText(tr("deselect_all"))
        if hasattr(self, 'apply_to_selected_btn'):
            self.apply_to_selected_btn.setText(tr("apply_to_selected"))
        if hasattr(self, 'apply_to_all_btn'):
            self.apply_to_all_btn.setText(tr("apply_to_all"))
            
        # Update rotate buttons in slice controls
        if hasattr(self, 'slice_controls'):
            for name, controls in self.slice_controls.items():
                if 'rotate_btn' in controls:
                    controls['rotate_btn'].setText(tr("rotate_90"))
                if 'slice_label' in controls:
                    controls['slice_label'].setText(tr("slice"))
                    
        # Update measurement settings labels
        if hasattr(self, 'measurements_label'):
            self.measurements_label.setText(tr("measurements"))
        if hasattr(self, 'line_width_label'):
            self.line_width_label.setText(tr("line_width"))
        if hasattr(self, 'line_color_label'):
            self.line_color_label.setText(tr("line_color"))
        if hasattr(self, 'text_color_label'):
            self.text_color_label.setText(tr("text_color"))
        if hasattr(self, 'font_size_label'):
            self.font_size_label.setText(tr("font_size"))
        if hasattr(self, 'font_weight_label'):
            self.font_weight_label.setText(tr("font_weight"))
        if hasattr(self, 'auto_snap_label'):
            self.auto_snap_label.setText(tr("auto_snap"))
        if hasattr(self, 'snap_distance_label'):
            self.snap_distance_label.setText(tr("snap_distance"))
                    
        # Update alpha label
        if hasattr(self, 'alpha_label'):
            current_alpha = self.alpha_slider.value() / 100.0 if hasattr(self, 'alpha_slider') else 0.50
            alpha_text = tr("alpha").rstrip(':') + f": {current_alpha:.2f}"
            self.alpha_label.setText(alpha_text)
            
        # Update font weight combo
        if hasattr(self, 'font_weight_combo'):
            current_data = [self.font_weight_combo.itemData(i) for i in range(self.font_weight_combo.count())]
            current_index = self.font_weight_combo.currentIndex()
            self.font_weight_combo.clear()
            self.font_weight_combo.addItem(tr("normal"), QFont.Normal)
            self.font_weight_combo.addItem(tr("bold"), QFont.Bold)
            if current_index < len(current_data):
                self.font_weight_combo.setCurrentIndex(current_index)
                
        # Update group box titles
        if hasattr(self, 'file_group'):
            self.file_group.setTitle(tr("file_operations"))
        if hasattr(self, 'view_group'):
            self.view_group.setTitle(tr("view_controls"))
        if hasattr(self, 'overlay_group'):
            self.overlay_group.setTitle(tr("overlay_controls"))
        if hasattr(self, 'label_colors_group'):
            self.label_colors_group.setTitle(tr("label_colors"))
        if hasattr(self, 'measure_group'):
            self.measure_group.setTitle(tr("measurement_tools"))
        if hasattr(self, 'settings_group'):
            self.settings_group.setTitle(tr("measurement_settings"))
        if hasattr(self, 'slice_groups'):
            for name, group in self.slice_groups.items():
                group.setTitle(f"{tr(name)} {tr('slice')}")

    def setup_ui(self) -> None:
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
        view_names = ["axial", "sagittal", "coronal"]
        view_titles = [tr("axial_view"), tr("sagittal_view"), tr("coronal_view")]

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

    def setup_control_dock(self) -> None:
        """Create docked control panel with collapsible sections."""
        self.control_dock = QDockWidget(tr("controls"), self)
        self.control_dock.setAllowedAreas(
            Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea
        )

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
        self.load_image_btn = QPushButton(tr("load_image_btn"))
        self.load_labels_btn = QPushButton(tr("load_labels_btn"))
        self.reset_btn = QPushButton(tr("reset_btn"))
        file_layout.addWidget(self.load_image_btn)
        file_layout.addWidget(self.load_labels_btn)
        file_layout.addWidget(self.reset_btn)
        self.image_path_label = QLabel(tr("image_path"))
        file_layout.addWidget(self.image_path_label)
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText(tr("no_image_loaded"))
        file_layout.addWidget(self.image_path_input)
        self.update_image_btn = QPushButton(tr("update_image"))
        file_layout.addWidget(self.update_image_btn)
        self.label_path_label = QLabel(tr("label_path"))
        file_layout.addWidget(self.label_path_label)
        self.label_path_input = QLineEdit()
        self.label_path_input.setPlaceholderText(tr("no_labels_loaded_placeholder"))
        file_layout.addWidget(self.label_path_input)
        self.update_labels_btn = QPushButton(tr("update_labels"))
        file_layout.addWidget(self.update_labels_btn)
        file_group = create_collapsible_group(tr("file_operations"), file_layout)
        self.file_group = file_group  # Store reference for language updates
        dock_layout.addWidget(file_group)

        # View controls
        view_layout = QGridLayout()
        self.view_checkboxes = {}
        view_names = ["axial", "sagittal", "coronal"]
        for i, name in enumerate(view_names):
            checkbox = QCheckBox(tr(name))
            checkbox.setChecked(True)
            self.view_checkboxes[name] = checkbox
            view_layout.addWidget(checkbox, 0, i)
        view_group = create_collapsible_group(tr("view_controls"), view_layout)
        self.view_group = view_group  # Store reference for language updates
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
            rotate_btn = QPushButton(tr("rotate_90"))
            slice_label = QLabel(tr("slice"))
            slice_layout.addWidget(slice_label)
            slice_layout.addWidget(slider)
            slice_layout.addWidget(spinbox)
            slice_layout.addWidget(rotate_btn)
            self.slice_controls[name] = {
                "slider": slider,
                "spinbox": spinbox,
                "rotate_btn": rotate_btn,
                "slice_label": slice_label,  # Store label reference
            }
            slice_group = create_collapsible_group(
                f"{tr(name)} {tr('slice')}", slice_layout
            )
            # Store references for language updates
            if not hasattr(self, 'slice_groups'):
                self.slice_groups = {}
            self.slice_groups[name] = slice_group
            dock_layout.addWidget(slice_group)

        # Overlay controls
        overlay_layout = QVBoxLayout()
        self.global_overlay_cb = QCheckBox(tr("show_overlay"))
        self.global_overlay_cb.setChecked(True)
        self.alpha_label = QLabel(tr("alpha_label"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        overlay_layout.addWidget(self.global_overlay_cb)
        overlay_layout.addWidget(self.alpha_label)
        overlay_layout.addWidget(self.alpha_slider)
        overlay_group = create_collapsible_group(
            tr("overlay_controls"), overlay_layout)
        self.overlay_group = overlay_group  # Store reference for language updates
        dock_layout.addWidget(overlay_group)

        # Label Colors controls
        self.label_colors_layout = QVBoxLayout()
        self.label_colors_scroll = QScrollArea()
        self.label_colors_scroll.setWidgetResizable(True)
        self.label_colors_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)
        self.label_colors_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        # Initial fixed height - will be dynamically adjusted when labels are loaded
        self.label_colors_scroll.setFixedHeight(80)

        self.label_colors_widget = QWidget()
        self.label_colors_content_layout = QVBoxLayout(
            self.label_colors_widget)
        self.label_colors_content_layout.setContentsMargins(4, 4, 4, 4)
        self.label_colors_content_layout.setSpacing(
            2
        )  # Tight spacing between label rows
        self.label_colors_scroll.setWidget(self.label_colors_widget)

        # Add initial empty label inside the scroll area
        empty_label = QLabel(tr("no_labels_loaded"))
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("color: #666; font-style: italic;")
        self.label_colors_content_layout.addWidget(empty_label)

        self.label_colors_layout.addWidget(self.label_colors_scroll)

        # Reset colors button
        self.reset_colors_btn = QPushButton(tr("reset_to_defaults"))
        self.reset_colors_btn.setEnabled(False)
        self.label_colors_layout.addWidget(self.reset_colors_btn)

        self.label_colors_group = create_collapsible_group(
            tr("label_colors"), self.label_colors_layout
        )
        self.label_colors_group.setChecked(False)  # Start collapsed
        dock_layout.addWidget(self.label_colors_group)

        # Measurement controls
        measure_layout = QVBoxLayout()

        # Tool buttons in horizontal layout
        tool_buttons_layout = QHBoxLayout()
        self.line_tool_btn = QPushButton(tr("line_tool"))
        self.line_tool_btn.setCheckable(True)
        self.eraser_tool_btn = QPushButton(tr("eraser"))
        self.eraser_tool_btn.setCheckable(True)
        tool_buttons_layout.addWidget(self.line_tool_btn)
        tool_buttons_layout.addWidget(self.eraser_tool_btn)
        measure_layout.addLayout(tool_buttons_layout)

        self.export_measurements_btn = QPushButton(tr("export_csv"))
        measure_layout.addWidget(self.export_measurements_btn)

        # Measurement Settings with Enhanced UI
        settings_layout = QVBoxLayout()

        # Measurement list with checkboxes for batch operations
        list_container_layout = QVBoxLayout()
        self.measurements_label = QLabel(tr("measurements"))
        list_container_layout.addWidget(self.measurements_label)
        self.measurements_list = QListWidget()
        self.measurements_list.setSelectionMode(QListWidget.MultiSelection)
        self.measurements_list.setContextMenuPolicy(Qt.CustomContextMenu)
        list_container_layout.addWidget(self.measurements_list)

        # Batch operation buttons
        batch_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton(tr("select_all"))
        self.select_all_btn.setMaximumWidth(80)
        self.deselect_all_btn = QPushButton(tr("deselect_all"))
        self.deselect_all_btn.setMaximumWidth(90)
        batch_buttons_layout.addWidget(self.select_all_btn)
        batch_buttons_layout.addWidget(self.deselect_all_btn)
        batch_buttons_layout.addStretch()
        list_container_layout.addLayout(batch_buttons_layout)

        settings_layout.addLayout(list_container_layout)

        # Style settings for selected measurements
        style_settings_layout = QGridLayout()
        self.line_width_label = QLabel(tr("line_width"))
        style_settings_layout.addWidget(self.line_width_label, 0, 0)
        self.line_width_spinbox = QDoubleSpinBox()
        self.line_width_spinbox.setRange(0.1, 10.0)  # 0.1 to 10.0 pixels
        self.line_width_spinbox.setValue(1.5)  # Default to 1.5 pixel (thinner)
        self.line_width_spinbox.setSingleStep(0.1)  # Step by 0.1
        self.line_width_spinbox.setDecimals(1)  # 1 decimal place
        style_settings_layout.addWidget(self.line_width_spinbox, 0, 1)

        self.line_color_label = QLabel(tr("line_color"))
        style_settings_layout.addWidget(self.line_color_label, 1, 0)
        self.line_color_btn = ColorButton(color=(255, 0, 0))  # Default red
        style_settings_layout.addWidget(self.line_color_btn, 1, 1)

        self.text_color_label = QLabel(tr("text_color"))
        style_settings_layout.addWidget(self.text_color_label, 2, 0)
        self.text_color_btn = ColorButton(color=(255, 0, 0))  # Default red
        style_settings_layout.addWidget(self.text_color_btn, 2, 1)

        self.font_size_label = QLabel(tr("font_size"))
        style_settings_layout.addWidget(self.font_size_label, 3, 0)
        self.font_size_spinbox = QDoubleSpinBox()
        self.font_size_spinbox.setRange(2.0, 24.0)  # 2.0 to 24.0 pt
        self.font_size_spinbox.setValue(5.0)  # Default to 5.0 pt (smaller)
        self.font_size_spinbox.setSingleStep(0.5)  # Step by 0.5
        self.font_size_spinbox.setDecimals(1)  # 1 decimal place
        style_settings_layout.addWidget(self.font_size_spinbox, 3, 1)

        self.font_weight_label = QLabel(tr("font_weight"))
        style_settings_layout.addWidget(self.font_weight_label, 4, 0)
        self.font_weight_combo = QComboBox()
        self.font_weight_combo.addItem(tr("normal"), QFont.Normal)
        self.font_weight_combo.addItem(tr("bold"), QFont.Bold)
        style_settings_layout.addWidget(self.font_weight_combo, 4, 1)

        settings_layout.addLayout(style_settings_layout)

        # Auto-snap settings
        snap_settings_layout = QGridLayout()
        self.auto_snap_label = QLabel(tr("auto_snap"))
        snap_settings_layout.addWidget(self.auto_snap_label, 0, 0)
        self.snap_enabled_checkbox = QCheckBox(tr("enable_endpoint_snapping"))
        self.snap_enabled_checkbox.setChecked(True)  # Default enabled
        snap_settings_layout.addWidget(self.snap_enabled_checkbox, 0, 1)

        self.snap_distance_label = QLabel(tr("snap_distance"))
        snap_settings_layout.addWidget(self.snap_distance_label, 1, 0)
        self.snap_distance_spinbox = QDoubleSpinBox()
        self.snap_distance_spinbox.setRange(
            3.0, 20.0)  # 3 to 20 pixels (smaller range)
        self.snap_distance_spinbox.setValue(
            12.0
        )  # Default 12 pixels for good snap feel
        self.snap_distance_spinbox.setSingleStep(1.0)
        self.snap_distance_spinbox.setDecimals(0)
        self.snap_distance_spinbox.setSuffix(" px")
        snap_settings_layout.addWidget(self.snap_distance_spinbox, 1, 1)

        settings_layout.addLayout(snap_settings_layout)

        # Apply buttons
        apply_buttons_layout = QHBoxLayout()
        self.apply_to_selected_btn = QPushButton(tr("apply_to_selected"))
        self.apply_to_all_btn = QPushButton(tr("apply_to_all"))
        apply_buttons_layout.addWidget(self.apply_to_selected_btn)
        apply_buttons_layout.addWidget(self.apply_to_all_btn)
        settings_layout.addLayout(apply_buttons_layout)

        measure_group = create_collapsible_group(
            tr("measurement_tools"), measure_layout)
        self.measure_group = measure_group  # Store reference for language updates
        dock_layout.addWidget(measure_group)

        settings_group = create_collapsible_group(
            tr("measurement_settings"), settings_layout
        )
        self.settings_group = settings_group  # Store reference for language updates
        # Start expanded to show the new features
        settings_group.setChecked(True)
        dock_layout.addWidget(settings_group)

        # Store color buttons for each label
        self.label_color_buttons = {}

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

    def setup_toolbar(self) -> None:
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
        self.panel_toggle_btn.setStyleSheet(
            """
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
        """
        )
        toolbar.addWidget(self.panel_toggle_btn)

        # Store toolbar reference
        self.main_toolbar = toolbar

    def setup_menu(self) -> None:
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu(tr("menu_file"))

        load_image_action = QAction(tr("load_image"), self)
        load_image_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_image_action)

        load_labels_action = QAction(tr("load_labels"), self)
        load_labels_action.setShortcut("Ctrl+L")
        file_menu.addAction(load_labels_action)

        save_submenu = file_menu.addMenu(tr("save"))
        save_image_action = QAction(tr("save_image"), self)
        save_label_action = QAction(tr("save_label"), self)
        save_overlay_action = QAction(tr("save_overlay"), self)
        save_submenu.addAction(save_image_action)
        save_submenu.addAction(save_label_action)
        save_submenu.addAction(save_overlay_action)

        # Shortcut to quickly open the Save submenu
        self.save_shortcut = QShortcut("Ctrl+Shift+S", self)
        self.save_shortcut.activated.connect(
            lambda: save_submenu.exec(QCursor.pos()))

        file_menu.addSeparator()

        save_screenshot_action = QAction(tr("save_screenshot"), self)
        save_screenshot_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_screenshot_action)

        file_menu.addSeparator()

        reset_action = QAction(tr("reset"), self)
        reset_action.setShortcut("Ctrl+R")
        file_menu.addAction(reset_action)

        file_menu.addSeparator()

        exit_action = QAction(tr("exit"), self)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu(tr("menu_view"))

        fit_all_action = QAction(tr("fit_all_views"), self)
        fit_all_action.setShortcut("F")
        view_menu.addAction(fit_all_action)

        view_menu.addSeparator()

        toggle_control_panel_action = QAction(tr("toggle_control_panel"), self)
        toggle_control_panel_action.setShortcut("Ctrl+T")
        toggle_control_panel_action.setCheckable(True)
        toggle_control_panel_action.setChecked(True)
        view_menu.addAction(toggle_control_panel_action)

        # Language menu
        language_menu = menubar.addMenu(tr("language"))
        
        # Create language action group for exclusive selection
        self.language_action_group = QActionGroup(self)
        
        english_action = QAction(tr("english"), self)
        english_action.setCheckable(True)
        english_action.setChecked(get_current_language() == "en")
        english_action.triggered.connect(lambda: self.change_language("en"))
        self.language_action_group.addAction(english_action)
        language_menu.addAction(english_action)
        
        chinese_action = QAction(tr("chinese"), self)
        chinese_action.setCheckable(True)
        chinese_action.setChecked(get_current_language() == "zh")
        chinese_action.triggered.connect(lambda: self.change_language("zh"))
        self.language_action_group.addAction(chinese_action)
        language_menu.addAction(chinese_action)
        
        french_action = QAction(tr("french"), self)
        french_action.setCheckable(True)
        french_action.setChecked(get_current_language() == "fr")
        french_action.triggered.connect(lambda: self.change_language("fr"))
        self.language_action_group.addAction(french_action)
        language_menu.addAction(french_action)

        # Help menu
        help_menu = menubar.addMenu(tr("menu_help"))

        about_action = QAction(tr("about"), self)
        about_action.setShortcut("F1")
        help_menu.addAction(about_action)

        # Store actions for controller access
        self.actions = {
            "load_image": load_image_action,
            "load_labels": load_labels_action,
            "save_image": save_image_action,
            "save_label": save_label_action,
            "save_overlay": save_overlay_action,
            "save_screenshot": save_screenshot_action,
            "reset": reset_action,
            "exit": exit_action,
            "fit_all": fit_all_action,
            "toggle_control_panel": toggle_control_panel_action,
            "about": about_action,
        }

    def change_language(self, language_code: str):
        """Change the application language and update UI."""
        set_language(language_code)
        self.update_ui_text()
        
        # Update language menu checkmarks
        for action in self.language_action_group.actions():
            if language_code == "en" and tr("english") in action.text():
                action.setChecked(True)
            elif language_code == "zh" and tr("chinese") in action.text():
                action.setChecked(True)
            elif language_code == "fr" and tr("french") in action.text():
                action.setChecked(True)
            else:
                action.setChecked(False)

    def setup_statusbar(self) -> None:
        """Create status bar."""
        self.status_bar = self.statusBar()
        self.status_label = QLabel(tr("status_ready"))
        self.status_bar.addWidget(self.status_label)

        # Coordinate display
        self.coord_label = QLabel(f"{tr('position')} -")
        self.status_bar.addPermanentWidget(self.coord_label)

    def keyPressEvent(self, event):
        """Handle key presses for deleting measurements."""
        if event.key() == Qt.Key_Delete:
            self.controller.delete_selected_measurement()
        super().keyPressEvent(event)


# ============================================================================
# CONTROLLER LAYER - Business Logic & Signal-Slot Binding
# ============================================================================


class ViewerController(QObject):
    """
    Main controller managing interactions between Model and View.
    Handles all user interactions, signal-slot connections, and business logic.
    """

    def __init__(self, model: ImageModel, preload_count: int = 2):
        super().__init__()

        self.model = model
        self.measurement_manager = MeasurementManager(model)
        self.thread_pool = QThreadPool()
        self.preload_thread_pool = QThreadPool()
        self.preload_thread_pool.setMaxThreadCount(2)  # Limit preload threads
        self.preload_count = preload_count
        self.default_output_path = None

        # Anti-bounce timer for smooth interactions
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_all_views)

        # Preload timer to avoid excessive preloading
        self.preload_timer = QTimer()
        self.preload_timer.setSingleShot(True)
        self.preload_timer.timeout.connect(self._preload_adjacent_slices)

    def set_view(self, view: MainWindow):
        self.view = view
        # Setup connections
        self.setup_model_connections()
        self.setup_view_connections()
        self.setup_slice_view_connections()
        self.setup_measurement_connections()

        # Connect models to slice views
        for slice_view in self.view.slice_views.values():
            slice_view.set_model(self.model)

    def setup_model_connections(self) -> None:
        """Connect model signals."""
        self.model.imageLoaded.connect(self._on_image_loaded)
        self.model.labelLoaded.connect(self._on_labels_loaded)
        self.model.loadError.connect(self._on_load_error)
        self.model.loadProgress.connect(self._on_load_progress)

    def setup_view_connections(self) -> None:
        """Connect main view signals."""
        # Menu actions
        self.view.actions["load_image"].triggered.connect(self.load_image)
        self.view.actions["load_labels"].triggered.connect(self.load_labels)
        self.view.actions["reset"].triggered.connect(self.reset_all)
        self.view.actions["exit"].triggered.connect(self.view.close)
        self.view.actions["fit_all"].triggered.connect(self.fit_all_views)
        self.view.actions["save_image"].triggered.connect(self._save_image)
        self.view.actions["save_label"].triggered.connect(self._save_label)
        self.view.actions["save_overlay"].triggered.connect(self._save_overlay)
        self.view.actions["save_screenshot"].triggered.connect(
            self.save_screenshot)
        self.view.actions["toggle_control_panel"].triggered.connect(
            self.toggle_control_panel
        )
        self.view.actions["about"].triggered.connect(self.show_about)

        # Control buttons
        self.view.load_image_btn.clicked.connect(self.load_image)
        self.view.load_labels_btn.clicked.connect(self.load_labels)
        self.view.reset_btn.clicked.connect(self.reset_all)

        # Path input buttons
        self.view.update_image_btn.clicked.connect(self.update_image_from_path)
        self.view.update_labels_btn.clicked.connect(
            self.update_labels_from_path)

        # View visibility
        for name, checkbox in self.view.view_checkboxes.items():
            checkbox.toggled.connect(
                partial(self._toggle_view_visibility, name))

        # Slice controls
        for name, controls in self.view.slice_controls.items():
            slider = controls["slider"]
            spinbox = controls["spinbox"]
            rotate_btn = controls["rotate_btn"]

            slider.valueChanged.connect(partial(self._on_slice_changed, name))
            spinbox.valueChanged.connect(partial(self._on_slice_changed, name))
            rotate_btn.clicked.connect(partial(self._rotate_view, name))

        # Overlay controls
        self.view.global_overlay_cb.toggled.connect(
            self._on_global_overlay_toggled)
        self.view.alpha_slider.valueChanged.connect(self._on_alpha_changed)

        # Label color controls
        self.view.reset_colors_btn.clicked.connect(self._reset_label_colors)

        # Control panel toggle button
        self.view.panel_toggle_btn.clicked.connect(self.toggle_control_panel)

    def setup_slice_view_connections(self) -> None:
        """Connect slice view signals."""
        for name, slice_view in self.view.slice_views.items():
            # Set controller reference for eraser functionality
            slice_view.controller = self
            slice_view.wheelScrolled.connect(
                partial(self._on_wheel_scroll, name))
            slice_view.mousePositionChanged.connect(
                partial(self._on_mouse_position, name)
            )
            slice_view.lineMeasured.connect(
                partial(self._on_line_measured, name))

            # Initialize snap settings from UI
            slice_view.snap_enabled = self.view.snap_enabled_checkbox.isChecked()
            slice_view.snap_distance = self.view.snap_distance_spinbox.value()

    def setup_measurement_connections(self):
        """Connect measurement-related signals."""
        self.view.line_tool_btn.clicked.connect(self.toggle_line_tool)
        self.view.eraser_tool_btn.clicked.connect(self.toggle_eraser_tool)
        self.measurement_manager.measurementAdded.connect(
            self._on_measurement_added)
        self.measurement_manager.measurementRemoved.connect(
            self._on_measurement_removed
        )
        self.measurement_manager.settingsChanged.connect(
            self._on_measurement_settings_changed
        )

        # Enhanced settings connections
        self.view.select_all_btn.clicked.connect(self.select_all_measurements)
        self.view.deselect_all_btn.clicked.connect(
            self.deselect_all_measurements)
        self.view.apply_to_selected_btn.clicked.connect(
            self.apply_style_to_selected)
        self.view.apply_to_all_btn.clicked.connect(self.apply_style_to_all)

        # Right-click context menu
        self.view.measurements_list.customContextMenuRequested.connect(
            self.show_measurement_context_menu
        )

        # Double-click to navigate to measurement
        self.view.measurements_list.itemDoubleClicked.connect(
            self.navigate_to_measurement
        )

        # Snap settings connections
        self.view.snap_enabled_checkbox.stateChanged.connect(
            self.update_snap_settings)
        self.view.snap_distance_spinbox.valueChanged.connect(
            self.update_snap_settings)

        # Export measurements
        self.view.export_measurements_btn.clicked.connect(self.export_measurements)

    def update_snap_settings(self):
        """Update snap settings for all slice views."""
        snap_enabled = self.view.snap_enabled_checkbox.isChecked()
        snap_distance = self.view.snap_distance_spinbox.value()

        for slice_view in self.view.slice_views.values():
            slice_view.snap_enabled = snap_enabled
            slice_view.snap_distance = snap_distance
            # Clear any existing snap preview when settings change
            slice_view._clear_snap_preview()
            slice_view._current_snap_point = None

    def toggle_line_tool(self, checked: bool):
        """Toggle the line measurement tool."""
        if checked:
            # Uncheck eraser tool
            self.view.eraser_tool_btn.blockSignals(True)
            self.view.eraser_tool_btn.setChecked(False)
            self.view.eraser_tool_btn.blockSignals(False)
            mode = "line"
        else:
            mode = "off"
        for view in self.view.slice_views.values():
            view.measure_mode = mode

    def toggle_eraser_tool(self, checked: bool):
        """Toggle the eraser tool."""
        if checked:
            # Uncheck line tool
            self.view.line_tool_btn.blockSignals(True)
            self.view.line_tool_btn.setChecked(False)
            self.view.line_tool_btn.blockSignals(False)
            mode = "erase"
        else:
            mode = "off"
        for view in self.view.slice_views.values():
            view.measure_mode = mode

    def select_all_measurements(self):
        """Select all measurements in the list."""
        for i in range(self.view.measurements_list.count()):
            item = self.view.measurements_list.item(i)
            if item:
                item.setSelected(True)

    def deselect_all_measurements(self):
        """Deselect all measurements in the list."""
        self.view.measurements_list.clearSelection()

    def apply_style_to_selected(self):
        """Apply current style settings to selected measurements."""
        selected_items = self.view.measurements_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self.view,
                "No Selection",
                "Please select measurements to apply style to.",
            )
            return

        measurement_ids = []
        for item in selected_items:
            measurement_id = item.data(Qt.UserRole)
            if measurement_id:
                measurement_ids.append(measurement_id)

        if measurement_ids:
            # Get current style settings from UI
            line_color = self.view.line_color_btn.color()
            line_width = self.view.line_width_spinbox.value()
            text_color = self.view.text_color_btn.color()
            text_font_size = self.view.font_size_spinbox.value()
            text_font_weight = self.view.font_weight_combo.currentData()

            # Apply to measurements
            self.measurement_manager.apply_style_to_measurements(
                measurement_ids,
                line_color=line_color,
                line_width=line_width,
                text_color=text_color,
                text_font_size=text_font_size,
                text_font_weight=text_font_weight,
            )

    def apply_style_to_all(self):
        """Apply current style settings to all measurements."""
        if not self.measurement_manager.measurements:
            QMessageBox.information(
                self.view, "No Measurements", "No measurements to apply style to."
            )
            return

        # Get current style settings from UI
        line_color = self.view.line_color_btn.color()
        line_width = self.view.line_width_spinbox.value()
        text_color = self.view.text_color_btn.color()
        text_font_size = self.view.font_size_spinbox.value()
        text_font_weight = self.view.font_weight_combo.currentData()

        # Apply to all measurements
        self.measurement_manager.apply_style_to_all_measurements(
            line_color=line_color,
            line_width=line_width,
            text_color=text_color,
            text_font_size=text_font_size,
            text_font_weight=text_font_weight,
        )

    def show_measurement_context_menu(self, position: QPoint):
        """Show context menu for measurement list."""
        item = self.view.measurements_list.itemAt(position)
        if not item:
            return

        # Create context menu
        menu = QMenu(self.view)

        # Copy coordinates action
        copy_coords_action = menu.addAction("Copy Coordinates")
        copy_coords_action.triggered.connect(
            lambda: self.copy_measurement_coordinates(item)
        )

        # Copy length action
        copy_length_action = menu.addAction("Copy Length")
        copy_length_action.triggered.connect(
            lambda: self.copy_measurement_length(item))

        menu.addSeparator()

        # Apply current style to all action
        apply_to_all_action = menu.addAction("Apply Style to All")
        apply_to_all_action.triggered.connect(self.apply_style_to_all)

        menu.addSeparator()

        # Delete action
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(
            lambda: self.delete_measurement_by_item(item))

        # Show menu
        menu.exec_(self.view.measurements_list.mapToGlobal(position))

    def copy_measurement_coordinates(self, item):
        """Copy measurement coordinates to clipboard."""
        measurement_id = item.data(Qt.UserRole)
        if measurement_id and measurement_id in self.measurement_manager.measurements:
            measurement = self.measurement_manager.measurements[measurement_id]
            coords_text = (
                f"Start: {measurement.start_world}\\n" f"End: {measurement.end_world}"
            )
            QApplication.clipboard().setText(coords_text)

    def copy_measurement_length(self, item):
        """Copy measurement length to clipboard."""
        measurement_id = item.data(Qt.UserRole)
        if measurement_id and measurement_id in self.measurement_manager.measurements:
            measurement = self.measurement_manager.measurements[measurement_id]
            QApplication.clipboard().setText(f"{measurement.length_mm:.2f} mm")

    def delete_measurement_by_item(self, item):
        """Delete measurement by list item."""
        measurement_id = item.data(Qt.UserRole)
        if measurement_id:
            self.measurement_manager.remove_measurement(measurement_id)

    def navigate_to_measurement(self, item):
        """Navigate to the slice containing the selected measurement."""
        measurement_id = item.data(Qt.UserRole)
        if measurement_id and measurement_id in self.measurement_manager.measurements:
            measurement = self.measurement_manager.measurements[measurement_id]
            # Navigate to the correct slice in the measurement's view
            self.model.set_slice(measurement.view_name, measurement.slice_idx)
            # Update the UI to reflect the new slice
            controls = self.view.slice_controls[measurement.view_name]
            controls["slider"].blockSignals(True)
            controls["spinbox"].blockSignals(True)
            controls["slider"].setValue(measurement.slice_idx)
            controls["spinbox"].setValue(measurement.slice_idx)
            controls["slider"].blockSignals(False)
            controls["spinbox"].blockSignals(False)
            # Trigger view update
            self._update_view(measurement.view_name)

    def _on_line_measured(self, view_name: str, start_pos: QPointF, end_pos: QPointF):
        """Handle a new line measurement from a slice view."""
        slice_idx = self.model.view_configs[view_name]["slice"]
        self.measurement_manager.add_line_measurement(
            view_name,
            slice_idx,
            (start_pos.x(), start_pos.y()),
            (end_pos.x(), end_pos.y()),
        )

    def _on_measurement_added(self, measurement: Measurement):
        """Add a graphical representation of the measurement to the scene."""
        self._create_measurement_graphics(measurement, measurement.view_name)

        # Trigger dynamic recalculation after adding the new measurement
        self._recalculate_text_positions(measurement.view_name)

        # Add to measurement list widget with enhanced display format
        display_text = (
            f"Line {measurement.id}: {measurement.length_mm:.2f} mm\n"
            f"  {measurement.view_name.title()} view, slice {measurement.slice_idx}"
        )

        # Create QListWidgetItem object and set data properly
        list_item = QListWidgetItem(display_text)
        list_item.setData(Qt.UserRole, measurement.id)

        # Add tooltip with detailed information
        tooltip_text = (
            f"Length: {measurement.length_mm:.2f} mm\n"
            f"View: {measurement.view_name.title()}\n"
            f"Slice: {measurement.slice_idx}\n"
            f"Start: ({measurement.start_world[0]:.1f}, {measurement.start_world[1]:.1f}, {measurement.start_world[2]:.1f})\n"
            f"End: ({measurement.end_world[0]:.1f}, {measurement.end_world[1]:.1f}, {measurement.end_world[2]:.1f})\n"
            f"Double-click to navigate to slice"
        )
        list_item.setToolTip(tooltip_text)

        # Add item to the list widget
        self.view.measurements_list.addItem(list_item)

    def _create_measurement_graphics(self, measurement: Measurement, view_name: str):
        """Create graphics items for a measurement in a specific view."""
        view = self.view.slice_views.get(view_name)
        if not view:
            return

        # Only show measurement on the correct slice
        current_slice = self.model.view_configs[view_name]["slice"]
        if current_slice != measurement.slice_idx:
            return

        axis = self.model.view_configs[view_name]["axis"]
        if axis == 0:  # Sagittal
            p1_img = QPointF(
                measurement.start_voxel[1], measurement.start_voxel[2])
            p2_img = QPointF(
                measurement.end_voxel[1], measurement.end_voxel[2])
        elif axis == 1:  # Coronal
            p1_img = QPointF(
                measurement.start_voxel[0], measurement.start_voxel[2])
            p2_img = QPointF(
                measurement.end_voxel[0], measurement.end_voxel[2])
        else:  # Axial
            p1_img = QPointF(
                measurement.start_voxel[0], measurement.start_voxel[1])
            p2_img = QPointF(
                measurement.end_voxel[0], measurement.end_voxel[1])

        # Apply the same Y-flip as during rendering for consistent display
        h = view.pixmap_item.pixmap().height()
        p1_draw = QPointF(p1_img.x(), h - 1 - p1_img.y())
        p2_draw = QPointF(p2_img.x(), h - 1 - p2_img.y())

        # Create draggable line graphics item as child of pixmap_item
        line_geom = QLineF(p1_draw.x(), p1_draw.y(), p2_draw.x(), p2_draw.y())
        line = DraggableLine(
            line_geom, measurement.id, self, view_name, parent=view.pixmap_item
        )
        pen = QPen(measurement.line_color, measurement.line_width)
        pen.setCosmetic(True)  # 关键：线宽按屏幕像素，不随缩放改变
        line.setPen(pen)
        line.setData(0, measurement.id)  # Store measurement ID
        line.setFlag(QGraphicsLineItem.ItemIsSelectable, True)
        line.setZValue(1)  # Base layer for lines

        # Create text graphics item as child of pixmap_item
        text = QGraphicsSimpleTextItem(
            f"{measurement.length_mm:.2f} mm", parent=view.pixmap_item
        )
        text.setBrush(QBrush(measurement.text_color))
        # Set font with custom size and weight
        font = text.font()
        # Support float font size
        font.setPointSizeF(measurement.text_font_size)
        font.setWeight(measurement.text_font_weight)
        text.setFont(font)

        # Text ignores transformations to stay upright and maintain constant size
        text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        # Use intelligent text positioning with collision avoidance and persistence
        line_center = QPointF(
            (p1_draw.x() + p2_draw.x()) / 2, (p1_draw.y() + p2_draw.y()) / 2
        )
        text_pos = self._calculate_text_position(
            view, p1_draw, p2_draw, text, measurement
        )
        text.setPos(text_pos)
        text.setData(0, measurement.id)
        text.setFlag(QGraphicsSimpleTextItem.ItemIsSelectable, True)
        text.setZValue(2)  # Text above lines

        # Add arrow only if text is significantly displaced from line segment
        arrow_item = None
        # Calculate distance from text center to line segment (screen-space threshold)
        scale = view.transform().m11()
        threshold_px = 18  # Screen pixels - only show arrow if text is far from line
        threshold_item = threshold_px / max(scale, 1e-6)

        distance_to_line = self._distance_point_to_segment(
            p1_draw, p2_draw, text_pos)
        if distance_to_line > threshold_item:
            arrow_item = self._create_arrow_annotation(
                text, p1_draw, p2_draw, measurement
            )

        # Create endpoint handles for dragging
        handle_radius = 5
        handle1 = EndpointHandle(
            p1_draw.x(),
            p1_draw.y(),
            handle_radius,
            measurement.id,
            0,
            self,
            view_name,
            parent=view.pixmap_item,
        )
        handle2 = EndpointHandle(
            p2_draw.x(),
            p2_draw.y(),
            handle_radius,
            measurement.id,
            1,
            self,
            view_name,
            parent=view.pixmap_item,
        )

        # Track graphics items (including arrow and handles)
        self.measurement_manager.add_graphics_items(
            measurement.id, view_name, line, text, arrow_item, handle1, handle2
        )

    def _calculate_text_position(
        self,
        view,
        p1: QPointF,
        p2: QPointF,
        text_item: QGraphicsSimpleTextItem,
        measurement=None,
        occupied_rects=None,
    ) -> QPointF:
        """Calculate optimal text position with stability and reservation system."""
        line_center = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)

        # Get text dimensions using font metrics (avoid scene manipulation)
        font = text_item.font()
        metrics = QFontMetricsF(font)
        text_width = metrics.horizontalAdvance(text_item.text())
        text_height = metrics.height()

        # Calculate line vectors
        line_vector = QPointF(p2.x() - p1.x(), p2.y() - p1.y())
        line_length = (line_vector.x() ** 2 + line_vector.y() ** 2) ** 0.5

        if line_length > 0:
            line_unit = QPointF(
                line_vector.x() / line_length, line_vector.y() / line_length
            )
            perp_unit = QPointF(-line_unit.y(), line_unit.x())
        else:
            line_unit = QPointF(1, 0)
            perp_unit = QPointF(0, 1)

        # Try to preserve existing position first
        if measurement and measurement.text_offset_img and measurement.text_anchor:
            old_pos = self._get_position_from_anchor(
                measurement.text_anchor,
                measurement.text_offset_img,
                p1,
                p2,
                line_center,
            )
            if old_pos and not self._has_collision_with_reservation(
                view,
                old_pos,
                text_width,
                text_height,
                text_item,
                p1,
                p2,
                occupied_rects,
            ):
                return old_pos

        # Define anchor points and their priorities
        anchor_points = [
            ("p1", p1, 1.0),  # Endpoint 1, highest priority
            ("p2", p2, 1.0),  # Endpoint 2, highest priority
            ("center", line_center, 0.8),  # Center, lower priority
        ]

        # Define offset directions with distances (极端防重叠策略)
        offset_close = max(10, text_height * 0.8)  # Close (increased again)
        # Medium distance (increased)
        offset_medium = max(20, text_height * 1.2)
        offset_far = max(35, text_height * 1.8)  # Far distance (increased)
        # Very far distance (increased)
        offset_very_far = max(50, text_height * 2.5)
        offset_extreme = max(
            70, text_height * 3.5
        )  # Extreme distance for dense areas (new)

        # Generate diverse candidates (法向 + 斜向 + 纯方向)
        directions = [
            (perp_unit.x(), perp_unit.y(), "right"),  # Right perpendicular
            (-perp_unit.x(), -perp_unit.y(), "left"),  # Left perpendicular
            # 斜向去同质化
            (
                0.7 * perp_unit.x() + 0.3 * line_unit.x(),
                0.7 * perp_unit.y() + 0.3 * line_unit.y(),
                "right_forward",
            ),
            (
                0.7 * perp_unit.x() - 0.3 * line_unit.x(),
                0.7 * perp_unit.y() - 0.3 * line_unit.y(),
                "right_back",
            ),
            (
                -0.7 * perp_unit.x() + 0.3 * line_unit.x(),
                -0.7 * perp_unit.y() + 0.3 * line_unit.y(),
                "left_forward",
            ),
            (
                -0.7 * perp_unit.x() - 0.3 * line_unit.x(),
                -0.7 * perp_unit.y() - 0.3 * line_unit.y(),
                "left_back",
            ),
            # 纯方向候选位置 (更好的分散性)
            (1.0, 0.0, "pure_right"),  # Pure horizontal right
            (-1.0, 0.0, "pure_left"),  # Pure horizontal left
            (0.0, 1.0, "pure_down"),  # Pure vertical down
            (0.0, -1.0, "pure_up"),  # Pure vertical up
        ]

        # Generate all candidates with priorities (极端防重叠优先级系统)
        candidates = []
        for anchor_name, anchor_pos, anchor_priority in anchor_points:
            for distance, dist_priority in [
                (offset_close, 5.0),
                (offset_medium, 4.0),
                (offset_far, 3.0),
                (offset_very_far, 2.0),
                (offset_extreme, 1.0),
            ]:
                for dx, dy, dir_name in directions:
                    pos = QPointF(
                        anchor_pos.x() + dx * distance, anchor_pos.y() + dy * distance
                    )

                    # Calculate priority: closer to line + endpoint preference
                    priority = anchor_priority * dist_priority
                    # Prefer positions similar to old position for stability
                    if measurement and measurement.text_offset_img:
                        old_offset = measurement.text_offset_img
                        new_offset = QPointF(dx * distance, dy * distance)
                        similarity = 1.0 / (
                            1.0
                            + (
                                (old_offset.x() - new_offset.x()) ** 2
                                + (old_offset.y() - new_offset.y()) ** 2
                            )
                            ** 0.5
                            / 20
                        )
                        priority *= (
                            1.0 + similarity * 0.5
                        )  # Bonus for similar positions

                    candidates.append(
                        (
                            pos,
                            priority,
                            anchor_name,
                            QPointF(dx * distance, dy * distance),
                        )
                    )

        # Sort by priority (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Add fallback center position
        candidates.append((line_center, 0.1, "center", QPointF(0, 0)))

        # Find best candidate
        best_pos = line_center
        best_anchor = "center"
        best_offset = QPointF(0, 0)

        for candidate_pos, _, anchor_name, offset in candidates:
            if not self._has_collision_with_reservation(
                view,
                candidate_pos,
                text_width,
                text_height,
                text_item,
                p1,
                p2,
                occupied_rects,
            ):
                best_pos = candidate_pos
                best_anchor = anchor_name
                best_offset = offset
                break

        # Store persistent layout info
        if measurement:
            measurement.text_anchor = best_anchor
            measurement.text_offset_img = best_offset

        return best_pos

    def _get_position_from_anchor(
        self, anchor: str, offset: QPointF, p1: QPointF, p2: QPointF, center: QPointF
    ) -> QPointF:
        """Restore position from anchor and offset."""
        if anchor == "p1":
            return QPointF(p1.x() + offset.x(), p1.y() + offset.y())
        elif anchor == "p2":
            return QPointF(p2.x() + offset.x(), p2.y() + offset.y())
        else:  # 'center'
            return QPointF(center.x() + offset.x(), center.y() + offset.y())

    def _has_collision_with_reservation(
        self,
        view,
        pos: QPointF,
        text_width: float,
        text_height: float,
        current_text: QGraphicsSimpleTextItem,
        line_p1: QPointF,
        line_p2: QPointF,
        occupied_rects=None,
    ) -> bool:
        """Enhanced collision detection with reservation system and boundary checking."""
        # Create text rect at position
        text_rect = QRectF(pos.x(), pos.y(), text_width, text_height)

        # Ultra-strong margin for dense measurement areas (zero-tolerance overlap)
        font_height = text_height
        collision_margin = max(
            16, font_height * 0.8
        )  # Further increased from 12 to 16, and 0.6 to 0.8
        expanded_rect = text_rect.adjusted(
            -collision_margin, -collision_margin, collision_margin, collision_margin
        )

        # Check image boundaries (avoid pushing text off-screen)
        if (
            hasattr(view, "pixmap_item")
            and view.pixmap_item.pixmap()
            and not view.pixmap_item.pixmap().isNull()
        ):
            pixmap = view.pixmap_item.pixmap()
            image_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
            margin_to_edge = 2  # Minimum distance to image edge
            safe_image_rect = image_rect.adjusted(
                margin_to_edge, margin_to_edge, -margin_to_edge, -margin_to_edge
            )

            # If text would be outside safe area, reject
            if not safe_image_rect.contains(text_rect):
                return True  # Collision = True means reject this position

        # Check reservation system first
        if occupied_rects:
            for occupied_rect in occupied_rects:
                if expanded_rect.intersects(occupied_rect):
                    return True

        # Check collision with existing measurement texts
        # 把 expanded_rect 先映射到 scene 再比较
        expanded_rect_scene = current_text.mapRectToScene(expanded_rect)
        for item in view.scene.items():
            if (
                isinstance(item, QGraphicsSimpleTextItem)
                and item != current_text
                and item.data(0) is not None  # Only check measurement texts
                and item.sceneBoundingRect().intersects(expanded_rect_scene)
            ):
                return True

        # Check collision with line
        return self._intersects_line(expanded_rect, line_p1, line_p2)

    def _has_collision(
        self,
        view,
        text_rect: QRectF,
        current_text: QGraphicsSimpleTextItem,
        line_p1: QPointF,
        line_p2: QPointF,
    ) -> bool:
        """Check if text collides with other texts or with its own line."""
        # Check collision with other texts
        if self._has_text_collision(view, text_rect, current_text):
            return True

        # Check collision with its own line
        if self._intersects_line(text_rect, line_p1, line_p2):
            return True

        return False

    def _intersects_line(self, rect: QRectF, p1: QPointF, p2: QPointF) -> bool:
        """Check if a rectangle intersects with a line segment."""
        # Expand rect slightly to avoid too-close positioning
        margin = 3  # Slightly reduced margin
        expanded_rect = rect.adjusted(-margin, -margin, margin, margin)

        # Check if either endpoint is inside the rectangle
        if expanded_rect.contains(p1) or expanded_rect.contains(p2):
            return True

        # More precise line-rectangle intersection check
        # Check if line segment intersects any of the rectangle's edges
        rect_left = expanded_rect.left()
        rect_right = expanded_rect.right()
        rect_top = expanded_rect.top()
        rect_bottom = expanded_rect.bottom()

        # Simple bounding box check first
        line_left = min(p1.x(), p2.x())
        line_right = max(p1.x(), p2.x())
        line_top = min(p1.y(), p2.y())
        line_bottom = max(p1.y(), p2.y())

        # If line's bounding box doesn't intersect rect, no collision
        if (
            line_right < rect_left
            or line_left > rect_right
            or line_bottom < rect_top
            or line_top > rect_bottom
        ):
            return False

        # If line passes through the center area, consider it a collision
        rect_center = expanded_rect.center()
        line_center = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)

        center_distance = (
            (rect_center.x() - line_center.x()) ** 2
            + (rect_center.y() - line_center.y()) ** 2
        ) ** 0.5

        # Use smaller diagonal threshold for more precise collision detection
        diagonal = ((expanded_rect.width()) ** 2 +
                    (expanded_rect.height()) ** 2) ** 0.5
        return center_distance < diagonal / 3  # More restrictive threshold

    def _has_text_collision(
        self, view, text_rect: QRectF, current_text: QGraphicsSimpleTextItem = None
    ) -> bool:
        """Check if a text rectangle collides with existing measurement texts."""
        collision_margin = (
            16  # Ultra-strong minimum spacing for dense areas (zero-tolerance)
        )
        expanded_rect = text_rect.adjusted(
            -collision_margin, -collision_margin, collision_margin, collision_margin
        )

        # Get all text items in the scene that have measurement IDs
        for item in view.scene.items():
            if (
                isinstance(item, QGraphicsSimpleTextItem)
                and item != current_text  # Exclude the current text item
                and item.data(0)
                is not None  # Only check measurement texts (they have IDs)
                and item.sceneBoundingRect().intersects(expanded_rect)
            ):
                return True

        return False

    def _closest_point_on_segment(
        self, p1: QPointF, p2: QPointF, point: QPointF
    ) -> QPointF:
        """Find the closest point on line segment p1-p2 to the given point."""
        # Vector from p1 to p2
        segment = QPointF(p2.x() - p1.x(), p2.y() - p1.y())
        # Vector from p1 to point
        to_point = QPointF(point.x() - p1.x(), point.y() - p1.y())

        # Length squared of segment
        segment_length_sq = segment.x() * segment.x() + segment.y() * segment.y()

        if segment_length_sq == 0:
            # p1 and p2 are the same point
            return p1

        # Project to_point onto segment
        t = (
            to_point.x() * segment.x() + to_point.y() * segment.y()
        ) / segment_length_sq

        # Clamp t to [0, 1] to stay within segment
        t = max(0.0, min(1.0, t))

        # Return the closest point on segment
        return QPointF(p1.x() + t * segment.x(), p1.y() + t * segment.y())

    def _distance_point_to_segment(
        self, p1: QPointF, p2: QPointF, point: QPointF
    ) -> float:
        """Calculate the shortest distance from a point to a line segment."""
        closest = self._closest_point_on_segment(p1, p2, point)
        dx = point.x() - closest.x()
        dy = point.y() - closest.y()
        return (dx * dx + dy * dy) ** 0.5

    def _create_arrow_path(self, text_item, p1_draw, p2_draw, measurement):
        """Create arrow path from closest point on line to text, with screen-space consistent geometry."""
        view = self.view.slice_views[measurement.view_name]
        pixmap_item = view.pixmap_item

        # 关键：用设备坐标得到真实绘制的文字矩形
        # 文字使用ItemIgnoresTransformations，实际渲染在设备空间，需要准确映射

        # 文本左上角场景坐标
        tl_scene = text_item.mapToScene(QPointF(0, 0))
        # 转到视图坐标（像素）
        tl_view = view.mapFromScene(tl_scene)

        # 用QFontMetricsF得到屏幕像素级的文字宽高
        metrics = QFontMetricsF(text_item.font())
        w = metrics.horizontalAdvance(text_item.text())
        h = metrics.height()

        # 视图坐标里的文字矩形 -> 场景坐标
        br_view = QPoint(int(tl_view.x() + w), int(tl_view.y() + h))
        tl_scene2 = view.mapToScene(QPoint(int(tl_view.x()), int(tl_view.y())))
        br_scene2 = view.mapToScene(br_view)
        scene_rect_visual = QRectF(tl_scene2, br_scene2).normalized()

        # 再把"真实矩形"映射到 pixmap_item 坐标（与线/箭头同系）
        text_rect = pixmap_item.mapRectFromScene(scene_rect_visual)
        text_center = text_rect.center()

        # 找到线段上离文字最近的点作为箭头起点（而不是固定的线段中心）
        closest_on_line = self._closest_point_on_segment(
            p1_draw, p2_draw, text_center)

        # 端点缓冲区：避免箭头从端点发射，造成视觉混乱
        seg = QPointF(p2_draw.x() - p1_draw.x(), p2_draw.y() - p1_draw.y())
        seg_len = (seg.x() ** 2 + seg.y() ** 2) ** 0.5

        if seg_len > 0:
            line_unit = QPointF(seg.x() / seg_len, seg.y() / seg_len)
            scale = view.transform().m11()
            margin_px = 15  # 端点缓冲（屏幕像素，确保箭头明显远离端点）
            margin_item = margin_px / max(scale, 1e-6)

            # 若落在端点缓冲内，往线段内部推开
            d1 = (
                (closest_on_line.x() - p1_draw.x()) ** 2
                + (closest_on_line.y() - p1_draw.y()) ** 2
            ) ** 0.5
            d2 = (
                (closest_on_line.x() - p2_draw.x()) ** 2
                + (closest_on_line.y() - p2_draw.y()) ** 2
            ) ** 0.5

            if d1 < margin_item and seg_len > 2 * margin_item:
                # 太靠近p1端点，往内部推开
                closest_on_line = QPointF(
                    p1_draw.x() + line_unit.x() * margin_item,
                    p1_draw.y() + line_unit.y() * margin_item,
                )
            elif d2 < margin_item and seg_len > 2 * margin_item:
                # 太靠近p2端点，往内部推开
                closest_on_line = QPointF(
                    p2_draw.x() - line_unit.x() * margin_item,
                    p2_draw.y() - line_unit.y() * margin_item,
                )
            elif seg_len <= 2 * margin_item:
                # 线段太短：用中点，避免贴端点
                closest_on_line = QPointF(
                    (p1_draw.x() + p2_draw.x()) /
                    2, (p1_draw.y() + p2_draw.y()) / 2
                )

        # 指向文字中心的单位向量
        v = QPointF(
            text_center.x() - closest_on_line.x(), text_center.y() - closest_on_line.y()
        )
        L = (v.x() ** 2 + v.y() ** 2) ** 0.5
        if L == 0:
            return QPainterPath()  # Empty path if no direction
        u = QPointF(v.x() / L, v.y() / L)

        # 与文字矩形求交，得到"触碰点"
        hit = self._intersect_ray_with_rect(closest_on_line, u, text_rect)

        # 屏幕像素等效的几何尺寸（随缩放自适应）
        scale = view.transform().m11()  # 当前缩放因子
        gap_px = 6  # 期望的屏幕像素间距
        head_len_px = 8  # 期望的屏幕像素箭头头长
        pad_px = 2  # 期望的屏幕像素退让距离

        # 转换为item坐标下的尺寸
        gap_item = gap_px / max(scale, 1e-6)
        head_len_item = head_len_px / max(scale, 1e-6)
        pad_item = pad_px / max(scale, 1e-6)

        # 起止点：从最近点出发，终点退让确保不穿过文字
        start = QPointF(
            closest_on_line.x() + u.x() * gap_item,
            closest_on_line.y() + u.y() * gap_item,
        )
        end = QPointF(
            hit.x() - u.x() * (head_len_item + pad_item),
            hit.y() - u.y() * (head_len_item + pad_item),
        )

        path = QPainterPath(start)
        path.lineTo(end)

        # 箭头三角（使用屏幕像素等效尺寸）
        perp = QPointF(-u.y(), u.x())
        head_w_px = 4  # 期望的屏幕像素箭头头部宽度
        head_w_item = head_w_px / max(scale, 1e-6)
        p1 = QPointF(
            end.x() - u.x() * head_len_item + perp.x() * head_w_item,
            end.y() - u.y() * head_len_item + perp.y() * head_w_item,
        )
        p2 = QPointF(
            end.x() - u.x() * head_len_item - perp.x() * head_w_item,
            end.y() - u.y() * head_len_item - perp.y() * head_w_item,
        )
        path.lineTo(p1)
        path.moveTo(end)
        path.lineTo(p2)

        return path

    def _create_arrow_annotation(self, text_item, p1_draw, p2_draw, measurement):
        """Create precise arrow pointing from line to text edge using geometric intersection."""
        path = self._create_arrow_path(
            text_item, p1_draw, p2_draw, measurement)
        if not path or path.isEmpty():
            return None

        item = QGraphicsPathItem(
            path, parent=text_item.parentItem())  # == pixmap_item
        pen = QPen(measurement.text_color, max(
            1.0, measurement.line_width * 0.9))
        pen.setCosmetic(True)  # 箭头恒定粗细，不随缩放改变
        item.setPen(pen)
        item.setZValue(1.5)  # Arrows between lines and text
        item.setData(0, measurement.id)
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        return item

    def _intersect_ray_with_rect(
        self, ray_origin: QPointF, ray_direction: QPointF, rect: QRectF
    ) -> QPointF:
        """Find intersection point of ray with rectangle edge (closest to origin)."""
        if rect.contains(ray_origin):
            # If origin is inside rect, return the center
            return rect.center()

        # Normalize ray direction
        ray_length = (ray_direction.x() ** 2 + ray_direction.y() ** 2) ** 0.5
        if ray_length == 0:
            return rect.center()

        ray_unit = QPointF(
            ray_direction.x() / ray_length, ray_direction.y() / ray_length
        )

        # Test intersection with each edge of the rectangle
        intersections = []

        # Left edge (x = rect.left())
        if ray_unit.x() != 0:
            t = (rect.left() - ray_origin.x()) / ray_unit.x()
            if t > 0:
                y = ray_origin.y() + t * ray_unit.y()
                if rect.top() <= y <= rect.bottom():
                    intersections.append(QPointF(rect.left(), y))

        # Right edge (x = rect.right())
        if ray_unit.x() != 0:
            t = (rect.right() - ray_origin.x()) / ray_unit.x()
            if t > 0:
                y = ray_origin.y() + t * ray_unit.y()
                if rect.top() <= y <= rect.bottom():
                    intersections.append(QPointF(rect.right(), y))

        # Top edge (y = rect.top())
        if ray_unit.y() != 0:
            t = (rect.top() - ray_origin.y()) / ray_unit.y()
            if t > 0:
                x = ray_origin.x() + t * ray_unit.x()
                if rect.left() <= x <= rect.right():
                    intersections.append(QPointF(x, rect.top()))

        # Bottom edge (y = rect.bottom())
        if ray_unit.y() != 0:
            t = (rect.bottom() - ray_origin.y()) / ray_unit.y()
            if t > 0:
                x = ray_origin.x() + t * ray_unit.x()
                if rect.left() <= x <= rect.right():
                    intersections.append(QPointF(x, rect.bottom()))

        # Return closest intersection point
        if intersections:
            closest = intersections[0]
            min_dist = (
                (closest.x() - ray_origin.x()) ** 2
                + (closest.y() - ray_origin.y()) ** 2
            ) ** 0.5
            for point in intersections[1:]:
                dist = (
                    (point.x() - ray_origin.x()) ** 2
                    + (point.y() - ray_origin.y()) ** 2
                ) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest = point
            return closest

        # Fallback to rect center
        return rect.center()

    def _intersect_ray_with_line(
        self,
        ray_origin: QPointF,
        ray_direction: QPointF,
        line_p1: QPointF,
        line_p2: QPointF,
    ) -> QPointF:
        """Find closest point on line segment to ray origin, or line center if ray doesn't intersect."""
        # Calculate closest point on line segment to ray origin
        line_vector = QPointF(line_p2.x() - line_p1.x(),
                              line_p2.y() - line_p1.y())
        line_length_sq = line_vector.x() ** 2 + line_vector.y() ** 2

        if line_length_sq == 0:
            return line_p1  # Line is a point

        # Project ray origin onto the line
        to_origin = QPointF(ray_origin.x() - line_p1.x(),
                            ray_origin.y() - line_p1.y())
        projection = (
            to_origin.x() * line_vector.x() + to_origin.y() * line_vector.y()
        ) / line_length_sq

        # Clamp to line segment
        projection = max(0.0, min(1.0, projection))

        closest_point = QPointF(
            line_p1.x() + projection * line_vector.x(),
            line_p1.y() + projection * line_vector.y(),
        )

        return closest_point

    def _recalculate_text_positions(self, view_name: str):
        """Smart recalculation that preserves good positions and only moves conflicting ones."""
        if view_name not in self.view.slice_views:
            return

        view = self.view.slice_views[view_name]
        current_slice = self.model.view_configs[view_name]["slice"]

        # Get all measurements for this view and slice
        measurements = []
        for measurement in self.measurement_manager.measurements.values():
            if (
                measurement.view_name == view_name
                and measurement.slice_idx == current_slice
                and measurement.id in self.measurement_manager._graphics_items
            ):
                measurements.append(measurement)

        if len(measurements) <= 1:
            return  # No need to recalculate for single measurement

        # Sort by creation order (older measurements get priority)
        measurements.sort(key=lambda m: m.id)

        # Step 1: Check which measurements have text collisions
        conflicting_measurements = set()
        for i, measurement1 in enumerate(measurements):
            graphics1 = self.measurement_manager._graphics_items[measurement1.id][
                view_name
            ]
            if len(graphics1) >= 2:
                text1 = graphics1[1]
                rect1 = text1.sceneBoundingRect()

                for j, measurement2 in enumerate(measurements[i + 1:], i + 1):
                    graphics2 = self.measurement_manager._graphics_items[
                        measurement2.id
                    ][view_name]
                    if len(graphics2) >= 2:
                        text2 = graphics2[1]
                        rect2 = text2.sceneBoundingRect()

                        # Check if texts overlap (with ultra-strong margin)
                        margin = 16
                        expanded_rect1 = rect1.adjusted(
                            -margin, -margin, margin, margin
                        )
                        if expanded_rect1.intersects(rect2):
                            conflicting_measurements.add(measurement1.id)
                            conflicting_measurements.add(measurement2.id)

        # Step 2: Build enhanced occupied_rects from non-conflicting measurements (comprehensive reservation)
        occupied_rects = []
        for measurement in measurements:
            if measurement.id not in conflicting_measurements:
                # Reserve space for good positions
                graphics = self.measurement_manager._graphics_items[measurement.id][
                    view_name
                ]
                if len(graphics) >= 2:
                    text_item = graphics[1]
                    text_rect = text_item.sceneBoundingRect()

                    # Enhanced adaptive margin based on font size and density
                    font_size = (
                        measurement.text_font_size
                        if hasattr(measurement, "text_font_size")
                        else 12
                    )
                    # Adaptive to font size
                    adaptive_margin = max(16, font_size * 1.2)

                    reserved_rect = text_rect.adjusted(
                        -adaptive_margin,
                        -adaptive_margin,
                        adaptive_margin,
                        adaptive_margin,
                    )
                    occupied_rects.append(reserved_rect)

                    # Also reserve space around arrows if they exist
                    if len(graphics) >= 3 and graphics[2]:  # Arrow exists
                        arrow_rect = graphics[2].sceneBoundingRect()
                        arrow_reserved = arrow_rect.adjusted(
                            -8, -8, 8, 8
                        )  # Smaller margin for arrows
                        occupied_rects.append(arrow_reserved)

        # Step 3: Recalculate positions for conflicting measurements with priority ordering
        # Sort conflicting measurements: locked first, then by age (older first)
        conflicting_list = [
            m for m in measurements if m.id in conflicting_measurements]
        conflicting_list.sort(
            key=lambda m: (not m.text_locked, m.id)
        )  # locked=True first, then by ID

        for measurement in conflicting_list:
            graphics_items = self.measurement_manager._graphics_items[measurement.id][
                view_name
            ]
            if len(graphics_items) >= 2:
                line_item = graphics_items[0]
                text_item = graphics_items[1]
                arrow_item = graphics_items[2] if len(
                    graphics_items) > 2 else None

                # Get line endpoints
                line = line_item.line()
                p1 = line.p1()
                p2 = line.p2()

                # Remove old arrow if it exists
                if arrow_item and arrow_item.scene():
                    view.scene.removeItem(arrow_item)

                # Recalculate text position with reservation system
                line_center = QPointF(
                    (p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
                new_text_pos = self._calculate_text_position(
                    view, p1, p2, text_item, measurement, occupied_rects
                )
                text_item.setPos(new_text_pos)

                # Reserve this new position for subsequent calculations
                font = text_item.font()
                metrics = QFontMetricsF(font)
                text_width = metrics.horizontalAdvance(text_item.text())
                text_height = metrics.height()
                text_rect = QRectF(
                    new_text_pos.x(), new_text_pos.y(), text_width, text_height
                )
                margin = max(6, text_height * 0.4)
                reserved_rect = text_rect.adjusted(
                    -margin, -margin, margin, margin)
                occupied_rects.append(reserved_rect)

                # Create new arrow if needed (only for far displacements from line segment)
                scale = view.transform().m11()
                threshold_px = 18  # Screen pixels - consistent with creation threshold
                threshold_item = threshold_px / max(scale, 1e-6)

                distance_to_line = self._distance_point_to_segment(
                    p1, p2, new_text_pos)
                new_arrow_item = None
                if distance_to_line > threshold_item:
                    new_arrow_item = self._create_arrow_annotation(
                        text_item, p1, p2, measurement
                    )

                # Update graphics items tracking
                self.measurement_manager.add_graphics_items(
                    measurement.id, view_name, line_item, text_item, new_arrow_item
                )

    def _on_measurement_settings_changed(self):
        """Redraw all measurements with new settings."""
        # Use the measurement manager's refresh method
        self.measurement_manager.refresh_measurement_display(self)

    def _on_measurement_removed(self, measurement_id: int):
        """Remove a measurement's graphics from all views."""
        # Remove from measurement list widget
        for i in range(self.view.measurements_list.count()):
            item = self.view.measurements_list.item(i)
            if item and item.data(Qt.UserRole) == measurement_id:
                self.view.measurements_list.takeItem(i)
                break

        # Graphics removal is handled by MeasurementManager.remove_measurement()

    def _on_handle_moved(
        self,
        view_name: str,
        measurement_id: int,
        handle_index: int,
        new_pos: QPointF,
        disable_snapping: bool = False,
    ):
        """Handle endpoint handle dragging with snapping and linked movement."""
        if measurement_id not in self.measurement_manager.measurements:
            return

        measurement = self.measurement_manager.measurements[measurement_id]
        view = self.view.slice_views[view_name]

        # Convert item coordinates back to image coordinates
        h = view.pixmap_item.pixmap().height()
        img_x = new_pos.x()
        img_y = h - 1 - new_pos.y()

        # Get current slice and axis
        current_slice = self.model.view_configs[view_name]["slice"]
        axis = self.model.view_configs[view_name]["axis"]

        # Unregister current endpoint from joints system
        old_joint_key = None
        for joint_key, endpoints in list(self.measurement_manager.joints.items()):
            if (measurement_id, handle_index) in endpoints:
                endpoints.discard((measurement_id, handle_index))
                if not endpoints:
                    del self.measurement_manager.joints[joint_key]
                else:
                    old_joint_key = joint_key
                break

        # Check UI snap enable setting
        if not view.snap_enabled:
            disable_snapping = True

        # Calculate screen-space constant snap distance
        scale = view.transform().m11()
        effective_snap_distance = self.measurement_manager.snap_distance / max(
            scale, 1e-6
        )

        # Check for snapping to existing joints (unless disabled by Ctrl key or UI setting)
        target_joint_key = None
        anchor_voxel_coords = None

        if not disable_snapping:
            nearby_joint = self.measurement_manager._find_nearby_joint(
                view_name, current_slice, (img_x,
                                           img_y), effective_snap_distance
            )
            if nearby_joint:
                # Snap to existing joint using anchor strategy
                target_joint_key = nearby_joint

                # Find an anchor point from existing joint members
                members = list(
                    self.measurement_manager.joints[target_joint_key])
                if members:
                    # Use the first existing member as anchor (prefer different measurement)
                    anchor_measurement_id, anchor_handle_index = members[0]
                    # Try to find a different measurement if possible
                    for mid, hidx in members:
                        if mid != measurement_id:
                            anchor_measurement_id, anchor_handle_index = mid, hidx
                            break

                    anchor_measurement = self.measurement_manager.measurements[
                        anchor_measurement_id
                    ]
                    if anchor_handle_index == 0:  # Start point
                        anchor_voxel_coords = anchor_measurement.start_voxel
                    else:  # End point
                        anchor_voxel_coords = anchor_measurement.end_voxel

        # If no snapping to existing joint, try endpoint-to-line-segment snapping
        if target_joint_key is None and not disable_snapping:
            closest_line_point = None
            min_line_distance = effective_snap_distance

            # Check all measurements on current slice (except current one)
            for other_measurement in self.measurement_manager.measurements.values():
                if (
                    other_measurement.id == measurement_id
                    or other_measurement.view_name != view_name
                    or other_measurement.slice_idx != current_slice
                ):
                    continue

                # Get line segment endpoints in image coordinates
                if axis == 0:  # Sagittal
                    p1_img = QPointF(
                        other_measurement.start_voxel[1],
                        other_measurement.start_voxel[2],
                    )
                    p2_img = QPointF(
                        other_measurement.end_voxel[1], other_measurement.end_voxel[2]
                    )
                elif axis == 1:  # Coronal
                    p1_img = QPointF(
                        other_measurement.start_voxel[0],
                        other_measurement.start_voxel[2],
                    )
                    p2_img = QPointF(
                        other_measurement.end_voxel[0], other_measurement.end_voxel[2]
                    )
                else:  # Axial
                    p1_img = QPointF(
                        other_measurement.start_voxel[0],
                        other_measurement.start_voxel[1],
                    )
                    p2_img = QPointF(
                        other_measurement.end_voxel[0], other_measurement.end_voxel[1]
                    )

                # Find closest point on line segment to current endpoint
                current_point = QPointF(img_x, img_y)
                closest_on_segment = self._intersect_ray_with_line(
                    current_point, QPointF(0, 0), p1_img, p2_img
                )

                # Calculate distance
                distance = (
                    (current_point.x() - closest_on_segment.x()) ** 2
                    + (current_point.y() - closest_on_segment.y()) ** 2
                ) ** 0.5

                if distance < min_line_distance:
                    min_line_distance = distance
                    closest_line_point = closest_on_segment

            # If found a close line segment, snap to it
            if closest_line_point is not None:
                # Convert closest point to voxel coordinates
                if axis == 0:  # Sagittal
                    anchor_voxel_coords = (
                        float(current_slice),
                        closest_line_point.x(),
                        closest_line_point.y(),
                    )
                elif axis == 1:  # Coronal
                    anchor_voxel_coords = (
                        closest_line_point.x(),
                        float(current_slice),
                        closest_line_point.y(),
                    )
                else:  # Axial
                    anchor_voxel_coords = (
                        closest_line_point.x(),
                        closest_line_point.y(),
                        float(current_slice),
                    )

                # Create joint at snapped position
                target_joint_key = self.measurement_manager._get_joint_key(
                    view_name,
                    current_slice,
                    (closest_line_point.x(), closest_line_point.y()),
                )

        # If still no snapping, create new joint at current position
        if target_joint_key is None:
            # Convert current position to voxel coordinates for new joint
            if axis == 0:  # Sagittal
                anchor_voxel_coords = (float(current_slice), img_x, img_y)
            elif axis == 1:  # Coronal
                anchor_voxel_coords = (img_x, float(current_slice), img_y)
            else:  # Axial
                anchor_voxel_coords = (img_x, img_y, float(current_slice))

            # Create new joint key for current position
            target_joint_key = self.measurement_manager._get_joint_key(
                view_name, current_slice, (img_x, img_y)
            )

        # Register current endpoint to target joint
        if target_joint_key not in self.measurement_manager.joints:
            self.measurement_manager.joints[target_joint_key] = set()
        self.measurement_manager.joints[target_joint_key].add(
            (measurement_id, handle_index)
        )

        # Update ONLY the current measurement endpoint to anchor position
        if handle_index == 0:  # Start point
            measurement.start_voxel = anchor_voxel_coords
            measurement.start_world = tuple(
                nib.affines.apply_affine(
                    self.model.image_affine, anchor_voxel_coords)
            )
        else:  # End point
            measurement.end_voxel = anchor_voxel_coords
            measurement.end_world = tuple(
                nib.affines.apply_affine(
                    self.model.image_affine, anchor_voxel_coords)
            )

        # Recalculate length for current measurement only
        measurement.length_mm = np.linalg.norm(
            np.array(measurement.start_world) - np.array(measurement.end_world)
        )

        # Update graphics for current measurement only
        self._update_measurement_graphics(measurement_id, view_name)

    def _on_line_moved(self, view_name: str, measurement_id: int, delta_item: QPointF):
        """Handle entire line dragging by updating measurement data with boundary constraints."""
        if measurement_id not in self.measurement_manager.measurements:
            return

        measurement = self.measurement_manager.measurements[measurement_id]
        view = self.view.slice_views[view_name]

        # Get image boundaries
        pixmap = view.pixmap_item.pixmap()
        if pixmap.isNull():
            return
        img_width, img_height = pixmap.width(), pixmap.height()

        # 先把 item 坐标系的位移向量"反旋转"到图像坐标系
        t_inv, _ = view.pixmap_item.transform().inverted()
        delta_img_vec = t_inv.map(delta_item)  # 纯旋转，无平移，这里等同于 mapVector

        # 图像是渲染时 flipud 的，所以再做一次 Y 翻转
        dx_img = delta_img_vec.x()
        dy_img = -delta_img_vec.y()

        # Convert current endpoints to image coordinates for boundary checking
        axis = self.model.view_configs[view_name]["axis"]
        if axis == 0:  # Sagittal
            p1_img = QPointF(
                measurement.start_voxel[1], measurement.start_voxel[2])
            p2_img = QPointF(
                measurement.end_voxel[1], measurement.end_voxel[2])
        elif axis == 1:  # Coronal
            p1_img = QPointF(
                measurement.start_voxel[0], measurement.start_voxel[2])
            p2_img = QPointF(
                measurement.end_voxel[0], measurement.end_voxel[2])
        else:  # Axial
            p1_img = QPointF(
                measurement.start_voxel[0], measurement.start_voxel[1])
            p2_img = QPointF(
                measurement.end_voxel[0], measurement.end_voxel[1])

        # Calculate constrained delta to keep both endpoints within bounds
        # Find maximum allowable delta in positive direction
        dx_max_pos = min(img_width - p1_img.x(), img_width - p2_img.x())
        dy_max_pos = min(img_height - p1_img.y(), img_height - p2_img.y())

        # Find maximum allowable delta in negative direction
        dx_max_neg = min(p1_img.x(), p2_img.x())
        dy_max_neg = min(p1_img.y(), p2_img.y())

        # Clamp delta to keep both points in bounds
        dx_clamped = max(-dx_max_neg, min(dx_max_pos, dx_img))
        dy_clamped = max(-dy_max_neg, min(dy_max_pos, dy_img))

        # Apply constrained delta to voxel coordinates
        def add_delta(voxel_coords):
            if axis == 0:  # Sagittal: (slice, x, y)
                return (
                    voxel_coords[0],
                    voxel_coords[1] + dx_clamped,
                    voxel_coords[2] + dy_clamped,
                )
            elif axis == 1:  # Coronal: (x, slice, y)
                return (
                    voxel_coords[0] + dx_clamped,
                    voxel_coords[1],
                    voxel_coords[2] + dy_clamped,
                )
            else:  # Axial: (x, y, slice)
                return (
                    voxel_coords[0] + dx_clamped,
                    voxel_coords[1] + dy_clamped,
                    voxel_coords[2],
                )

        # 1) 移动前：从 joints 移除本测量全部端点
        self.measurement_manager._unregister_measurement_from_joints(
            measurement_id)

        # Update voxel coordinates with consistent boundary constraints
        measurement.start_voxel = add_delta(measurement.start_voxel)
        measurement.end_voxel = add_delta(measurement.end_voxel)

        # Recalculate world coordinates and length
        measurement.start_world = tuple(
            nib.affines.apply_affine(
                self.model.image_affine, measurement.start_voxel)
        )
        measurement.end_world = tuple(
            nib.affines.apply_affine(
                self.model.image_affine, measurement.end_voxel)
        )
        measurement.length_mm = np.linalg.norm(
            np.array(measurement.start_world) - np.array(measurement.end_world)
        )

        # 3) 计算新的图像平面端点坐标（注意这里是"图像坐标"，非翻转后的绘制坐标）
        if axis == 0:  # sagittal
            p1_img_new = (
                measurement.start_voxel[1], measurement.start_voxel[2])
            p2_img_new = (measurement.end_voxel[1], measurement.end_voxel[2])
        elif axis == 1:  # coronal
            p1_img_new = (
                measurement.start_voxel[0], measurement.start_voxel[2])
            p2_img_new = (measurement.end_voxel[0], measurement.end_voxel[2])
        else:  # axial
            p1_img_new = (
                measurement.start_voxel[0], measurement.start_voxel[1])
            p2_img_new = (measurement.end_voxel[0], measurement.end_voxel[1])

        # 4) 按新位置把该测量重新注册到 joints
        current_slice = self.model.view_configs[view_name]["slice"]
        self.measurement_manager._register_measurement_to_joints(
            measurement_id, view_name, current_slice, p1_img_new, p2_img_new
        )

        # Update graphics using unified data-driven approach
        self._update_measurement_graphics(measurement_id, view_name)

    def _update_measurement_graphics(self, measurement_id: int, view_name: str):
        """Update graphics after measurement data change."""
        if measurement_id not in self.measurement_manager.measurements:
            return

        measurement = self.measurement_manager.measurements[measurement_id]
        view = self.view.slice_views[view_name]

        # Get current graphics items
        if view_name not in self.measurement_manager._graphics_items.get(
            measurement_id, {}
        ):
            return

        graphics_items = self.measurement_manager._graphics_items[measurement_id][
            view_name
        ]
        if len(graphics_items) < 2:
            return

        line_item = graphics_items[0]
        text_item = graphics_items[1]

        # Convert voxel coordinates to draw coordinates
        axis = self.model.view_configs[view_name]["axis"]
        if axis == 0:  # Sagittal
            p1_img = QPointF(
                measurement.start_voxel[1], measurement.start_voxel[2])
            p2_img = QPointF(
                measurement.end_voxel[1], measurement.end_voxel[2])
        elif axis == 1:  # Coronal
            p1_img = QPointF(
                measurement.start_voxel[0], measurement.start_voxel[2])
            p2_img = QPointF(
                measurement.end_voxel[0], measurement.end_voxel[2])
        else:  # Axial
            p1_img = QPointF(
                measurement.start_voxel[0], measurement.start_voxel[1])
            p2_img = QPointF(
                measurement.end_voxel[0], measurement.end_voxel[1])

        # Apply Y-flip for drawing
        h = view.pixmap_item.pixmap().height()
        p1_draw = QPointF(p1_img.x(), h - 1 - p1_img.y())
        p2_draw = QPointF(p2_img.x(), h - 1 - p2_img.y())

        # Update line geometry
        line_item.setLine(p1_draw.x(), p1_draw.y(), p2_draw.x(), p2_draw.y())

        # Update handles positions (if they exist)
        if len(graphics_items) > 3:  # Has handles
            handle1 = graphics_items[3]
            if handle1:
                handle1.setPos(p1_draw)
            if len(graphics_items) > 4 and graphics_items[4]:
                handle2 = graphics_items[4]
                handle2.setPos(p2_draw)

        # Update text content and position
        text_item.setText(f"{measurement.length_mm:.2f} mm")
        # Ensure text ignores transformations to stay upright and maintain constant size
        text_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        line_center = QPointF(
            (p1_draw.x() + p2_draw.x()) / 2, (p1_draw.y() + p2_draw.y()) / 2
        )
        new_text_pos = self._calculate_text_position(
            view, p1_draw, p2_draw, text_item, measurement
        )
        text_item.setPos(new_text_pos)

        # Update or create arrow if needed - safe array access
        arrow_item = None
        if len(graphics_items) > 2 and graphics_items[2]:
            arrow_item = graphics_items[2]

        # Check if arrow is needed based on distance to line segment (screen-space threshold)
        scale = view.transform().m11()
        threshold_px = 18  # Screen pixels - consistent with other thresholds
        threshold_item = threshold_px / max(scale, 1e-6)

        distance_to_line = self._distance_point_to_segment(
            p1_draw, p2_draw, new_text_pos
        )

        # Create new arrow path if needed
        new_path = None
        if distance_to_line > threshold_item:
            new_path = self._create_arrow_path(
                text_item, p1_draw, p2_draw, measurement)

        # Update existing arrow or create new one
        if len(graphics_items) > 2 and graphics_items[2]:
            if new_path is None or new_path.isEmpty():
                # Have old arrow but new rules don't need it -> remove and clear
                if arrow_item.scene():
                    arrow_item.scene().removeItem(arrow_item)
                graphics_items[2] = None
            else:
                # Update existing arrow path (avoid flicker)
                arrow_item.setPath(new_path)
                # Update color in case measurement style changed
                pen = QPen(
                    measurement.text_color, max(
                        1.0, measurement.line_width * 0.9)
                )
                pen.setCosmetic(True)
                arrow_item.setPen(pen)
        else:
            if new_path is not None and not new_path.isEmpty():
                # Create new arrow
                new_arrow_item = self._create_arrow_annotation(
                    text_item, p1_draw, p2_draw, measurement
                )
                # Ensure graphics_items has enough elements and backfill arrow at index 2
                while len(graphics_items) <= 2:
                    graphics_items.append(None)
                graphics_items[2] = new_arrow_item

    def delete_selected_measurement(self):
        """Delete the currently selected measurement."""
        # Check list widget selection first
        selected_list_items = self.view.measurements_list.selectedItems()
        if selected_list_items:
            measurement_id = selected_list_items[0].data(Qt.UserRole)
            self.measurement_manager.remove_measurement(measurement_id)
            return

        for view in self.view.slice_views.values():
            selected_items = view.scene.selectedItems()
            if selected_items:
                # Assuming the first selected item's data is the measurement ID
                measurement_id = selected_items[0].data(0)
                if measurement_id:
                    self.measurement_manager.remove_measurement(measurement_id)
                    break  # Assume only one measurement can be selected at a time

    def export_measurements(self):
        """Export measurements to a CSV file."""
        if not self.measurement_manager.measurements:
            QMessageBox.warning(self.view, "Warning",
                                "No measurements to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self.view, "Export Measurements", "measurements.csv", "CSV Files (*.csv)"
        )

        if filepath:
            try:
                self.measurement_manager.export_csv(filepath)
                self.view.status_label.setText(
                    f"Measurements exported to {Path(filepath).name}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self.view, "Error", f"Failed to export measurements: {e}"
                )

    def load_image(self) -> None:
        """Load image file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self.view,
            "Load Medical Image",
            "",
            "Image Files (*.nii *.nii.gz *.mha *.mhd);;All Files (*)",
        )

        if filepath:
            self.view.status_label.setText(tr("status_loading_image"))

            # Show progress bar if control panel exists
            if (
                hasattr(self.view, "control_dock")
                and self.view.control_dock is not None
            ):
                self.view.progress_bar.setValue(0)
                self.view.progress_bar.setVisible(True)

            # Load in background
            worker = LoadWorker(self.model, filepath, False)
            self.thread_pool.start(worker)

    def load_labels(self) -> None:
        """Load labels file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self.view,
            "Load Medical Labels",
            "",
            "Label Files (*.nii *.nii.gz *.mha *.mhd);;All Files (*)",
        )

        if filepath:
            self.view.status_label.setText(tr("status_loading_labels"))

            # Show progress bar if control panel exists
            if (
                hasattr(self.view, "control_dock")
                and self.view.control_dock is not None
            ):
                self.view.progress_bar.setValue(0)
                self.view.progress_bar.setVisible(True)

            # Load in background
            worker = LoadWorker(self.model, filepath, True)
            self.thread_pool.start(worker)

    def update_image_from_path(self) -> None:
        """Load image from path input field."""
        # Check if control panel exists
        if not (
            hasattr(self.view, "control_dock") and self.view.control_dock is not None
        ):
            return

        filepath = self.view.image_path_input.text().strip()
        if not filepath:
            QMessageBox.warning(self.view, "Warning",
                                "Please enter an image file path")
            return

        if not Path(filepath).exists():
            QMessageBox.warning(self.view, "Warning",
                                f"File not found: {filepath}")
            return

        self.view.status_label.setText(tr("status_loading_image_path"))
        self.view.progress_bar.setValue(0)
        self.view.progress_bar.setVisible(True)

        # Load in background
        worker = LoadWorker(self.model, filepath, False)
        self.thread_pool.start(worker)

    def update_labels_from_path(self) -> None:
        """Load labels from path input field."""
        # Check if control panel exists
        if not (
            hasattr(self.view, "control_dock") and self.view.control_dock is not None
        ):
            return

        filepath = self.view.label_path_input.text().strip()
        if not filepath:
            QMessageBox.warning(self.view, "Warning",
                                "Please enter a label file path")
            return

        if not Path(filepath).exists():
            QMessageBox.warning(self.view, "Warning",
                                f"File not found: {filepath}")
            return

        self.view.status_label.setText(tr("status_loading_labels_path"))
        self.view.progress_bar.setValue(0)
        self.view.progress_bar.setVisible(True)

        # Load in background
        worker = LoadWorker(self.model, filepath, True)
        self.thread_pool.start(worker)

    def reset_all(self) -> None:
        """Reset all views and data."""
        # Clear model
        self.model.image_data = None
        self.model.label_data = None
        self.model.get_slice_data.cache_clear()
        self.model._render_slice_cached.cache_clear()
        QPixmapCache.clear()

        # Reset UI - check if control panel exists
        if hasattr(self.view, "control_dock") and self.view.control_dock is not None:
            for controls in self.view.slice_controls.values():
                controls["slider"].setValue(0)
                controls["spinbox"].setValue(0)

            # Clear path inputs
            self.view.image_path_input.clear()
            self.view.label_path_input.clear()

            # Clear label color controls
            self._clear_label_color_controls()

        for slice_view in self.view.slice_views.values():
            slice_view.scene.clear()
            slice_view.pixmap_item = QGraphicsPixmapItem()
            slice_view.scene.addItem(slice_view.pixmap_item)

        self.view.status_label.setText(tr("status_ready"))

        # Reset progress bar if control panel exists
        if hasattr(self.view, "control_dock") and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)

    def fit_all_views(self) -> None:
        """Fit all views to show full image."""
        for slice_view in self.view.slice_views.values():
            slice_view.reset_view()

    def _save_image(self) -> None:
        if self.model.image_data is None:
            QMessageBox.warning(self.view, "Warning", "No image loaded.")
            return
        filters = "NIfTI (*.nii *.nii.gz);;MetaImage (*.mha *.mhd);;All Files (*)"
        default = (
            self.model.strip_extensions(self.model.image_path)
            if self.model.image_path
            else ""
        )
        path, _ = QFileDialog.getSaveFileName(
            self.view, "Save Image Only", default, filters
        )
        if path:
            try:
                path = self.model.save_volume(
                    self.model.image_data, self.model.image_path, path
                )
                self.view.status_label.setText(
                    f"Image saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))

    def _save_label(self) -> None:
        if self.model.label_data is None:
            QMessageBox.warning(self.view, "Warning", "No label loaded.")
            return
        filters = "NIfTI (*.nii *.nii.gz);;MetaImage (*.mha *.mhd);;All Files (*)"
        default = (
            self.model.strip_extensions(self.model.label_path)
            if self.model.label_path
            else ""
        )
        path, _ = QFileDialog.getSaveFileName(
            self.view, "Save Label Only", default, filters
        )
        if path:
            try:
                path = self.model.save_volume(
                    self.model.label_data, self.model.label_path, path
                )
                self.view.status_label.setText(
                    f"Label saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))

    def _save_overlay(self) -> None:
        if self.model.image_data is None or self.model.label_data is None:
            QMessageBox.warning(
                self.view, "Warning", "Need both image and label loaded."
            )
            return
        filters = "NIfTI (*.nii *.nii.gz);;MetaImage (*.mha *.mhd);;All Files (*)"
        default = (
            self.model.strip_extensions(self.model.image_path)
            if self.model.image_path
            else ""
        )
        path, _ = QFileDialog.getSaveFileName(
            self.view, "Save Overlay Image", default, filters
        )
        if path:
            try:
                overlay_vol = self.model.make_overlay_volume()
                path = self.model.save_volume(
                    overlay_vol, self.model.image_path, path)
                self.view.status_label.setText(
                    f"Overlay saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))

    def save_screenshot(self) -> None:
        """Save current view as screenshot."""
        default_name = self.default_output_path or "screenshot.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self.view,
            "Save Screenshot",
            default_name,
            "PNG Files (*.png);;All Files (*)",
        )

        if filepath:
            # Capture the central widget
            pixmap = self.view.centralWidget().grab()
            pixmap.save(filepath)
            self.view.status_label.setText(
                f"Screenshot saved: {Path(filepath).name}")

    def show_about(self) -> None:
        """Show About dialog."""
        about_dialog = AboutDialog(self.view)
        about_dialog.exec()

    def toggle_control_panel(self) -> None:
        """Toggle control panel visibility - completely remove/add dock."""
        is_visible = (
            hasattr(self.view, "control_dock") and self.view.control_dock is not None
        )

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
            new_width = max(
                current_size.width() - dock_width - 10, 600
            )  # Minimum 600px
            self.view.resize(new_width, current_size.height())

            # Update button appearance
            self.view.panel_toggle_btn.setText("▶")
            self.view.panel_toggle_btn.setToolTip(
                "Show Control Panel (Ctrl+T)")
            self.view.actions["toggle_control_panel"].setChecked(False)

        else:
            # Recreate the control dock
            self.view.setup_control_dock()

            # Reconnect the control panel signals
            self._reconnect_control_signals()

            # Restore window width
            if hasattr(self.view, "_stored_window_width"):
                current_size = self.view.size()
                self.view.resize(self.view._stored_window_width,
                                 current_size.height())
            else:
                # Default expanded width
                current_size = self.view.size()
                self.view.resize(current_size.width() +
                                 300, current_size.height())

            # Update button appearance
            self.view.panel_toggle_btn.setText("◀")
            self.view.panel_toggle_btn.setToolTip(
                "Hide Control Panel (Ctrl+T)")
            self.view.actions["toggle_control_panel"].setChecked(True)

    def _reconnect_control_signals(self) -> None:
        """Reconnect control panel signals after recreating the dock."""
        # Control buttons
        self.view.load_image_btn.clicked.connect(self.load_image)
        self.view.load_labels_btn.clicked.connect(self.load_labels)
        self.view.reset_btn.clicked.connect(self.reset_all)

        # Path input buttons
        self.view.update_image_btn.clicked.connect(self.update_image_from_path)
        self.view.update_labels_btn.clicked.connect(
            self.update_labels_from_path)

        # View visibility
        for name, checkbox in self.view.view_checkboxes.items():
            checkbox.toggled.connect(
                partial(self._toggle_view_visibility, name))

        # Slice controls
        for name, controls in self.view.slice_controls.items():
            slider = controls["slider"]
            spinbox = controls["spinbox"]
            rotate_btn = controls["rotate_btn"]

            slider.valueChanged.connect(partial(self._on_slice_changed, name))
            spinbox.valueChanged.connect(partial(self._on_slice_changed, name))
            rotate_btn.clicked.connect(partial(self._rotate_view, name))

        # Overlay controls
        self.view.global_overlay_cb.toggled.connect(
            self._on_global_overlay_toggled)
        self.view.alpha_slider.valueChanged.connect(self._on_alpha_changed)

    def _on_image_loaded(
        self, filepath: str, shape: tuple, affine: np.ndarray, spacing: tuple
    ) -> None:
        """Handle successful image loading."""
        self.view.status_label.setText(
            f"Image loaded: {Path(filepath).name} - Shape: {shape}"
        )

        # Update control panel elements if they exist
        if hasattr(self.view, "control_dock") and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)

            # Update image path input
            self.view.image_path_input.setText(filepath)

            # Update slice controls
            for name, config in self.model.view_configs.items():
                axis = config["axis"]
                max_slice = shape[axis] - 1
                mid_slice = max_slice // 2

                controls = self.view.slice_controls[name]
                controls["slider"].setMaximum(max_slice)
                controls["spinbox"].setMaximum(max_slice)
                controls["slider"].setValue(mid_slice)
                controls["spinbox"].setValue(mid_slice)

                self.model.set_slice(name, mid_slice)
        else:
            # Set model slice positions directly when no controls exist
            for name, config in self.model.view_configs.items():
                axis = config["axis"]
                mid_slice = shape[axis] // 2
                self.model.set_slice(name, mid_slice)

        self._update_all_views()

    def _on_labels_loaded(self, filepath: str, unique_count: int) -> None:
        """Handle successful label loading."""
        self.view.status_label.setText(
            f"Labels loaded: {Path(filepath).name} - {unique_count} unique labels"
        )

        # Update control panel elements if they exist
        if hasattr(self.view, "control_dock") and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)
            # Update label path input
            self.view.label_path_input.setText(filepath)
            # Populate color controls
            self._populate_label_color_controls()

        self._update_all_views()

    def _on_load_error(self, error_msg: str) -> None:
        """Handle loading errors."""
        self.view.status_label.setText(tr("status_load_failed"))

        # Hide progress bar if control panel exists
        if hasattr(self.view, "control_dock") and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)

        QMessageBox.critical(self.view, "Load Error", error_msg)

    def _on_load_progress(self, value: int) -> None:
        """Update progress bar during loading."""
        if hasattr(self.view, "control_dock") and self.view.control_dock is not None:
            self.view.progress_bar.setValue(value)
            if value >= 100:
                self.view.progress_bar.setVisible(False)

    def _toggle_view_visibility(self, view_name: str, visible: bool) -> None:
        """Toggle view visibility."""
        self.model.view_configs[view_name]["show"] = visible

        # Find the view frame and hide/show it
        view_index = ["axial", "sagittal", "coronal"].index(view_name)
        widget = self.view.splitter.widget(view_index)
        widget.setVisible(visible)

    def _on_slice_changed(self, view_name: str, value: int) -> None:
        """Handle slice navigation."""
        # Sync slider and spinbox
        controls = self.view.slice_controls[view_name]
        controls["slider"].blockSignals(True)
        controls["spinbox"].blockSignals(True)
        controls["slider"].setValue(value)
        controls["spinbox"].setValue(value)
        controls["slider"].blockSignals(False)
        controls["spinbox"].blockSignals(False)

        # Update model
        self.model.set_slice(view_name, value)

        # Update measurement display for this slice
        if hasattr(self, "measurement_manager"):
            self.measurement_manager.update_measurement_display(
                view_name, value, self)

        # Debounced update
        self.update_timer.start(40)  # 40ms debounce for smooth scrolling

        # Show/hide measurements
        for mid, measurement in self.measurement_manager.measurements.items():
            for item in self.view.slice_views[view_name].scene.items():
                if item.data(0) == mid:
                    is_visible = (
                        measurement.view_name == view_name
                        and measurement.slice_idx == value
                    )
                    item.setVisible(is_visible)

    def _on_wheel_scroll(self, view_name: str, delta: int) -> None:
        """Handle mouse wheel slice navigation."""
        controls = self.view.slice_controls[view_name]
        current_value = controls["slider"].value()
        new_value = current_value + delta

        # Clamp to valid range
        max_value = controls["slider"].maximum()
        new_value = max(0, min(new_value, max_value))

        self._on_slice_changed(view_name, new_value)

    def _on_mouse_position(self, view_name: str, x: int, y: int) -> None:
        """Handle mouse position for tooltips and coordinates."""
        # Update coordinate display
        self.view.coord_label.setText(f"{tr('position')} ({x}, {y})")

        # Show label tooltip if available
        if self.model.label_data is not None:
            label_value = self.model.get_label_at_position(view_name, x, y)
            if label_value > 0:
                QToolTip.showText(
                    self.view.slice_views[view_name].mapToGlobal(
                        self.view.slice_views[view_name].mapFromScene(
                            QPointF(x, y))
                    ),
                    f"Label: {label_value}",
                )

    def _rotate_view(self, view_name: str) -> None:
        """Rotate view by 90 degrees."""
        config = self.model.view_configs[view_name]
        config["rotation"] = (config["rotation"] + 90) % 360
        self._update_view(view_name)
        # Auto fit after rotation to avoid manual F key press
        self.view.slice_views[view_name].reset_view()

        # Force recalculate measurements for this view after rotation
        # This ensures text positions and arrow geometry are updated for the new rotation
        slice_idx = self.model.view_configs[view_name]["slice"]
        self.measurement_manager.update_measurement_display(
            view_name, slice_idx, self)

    def _on_global_overlay_toggled(self, checked: bool) -> None:
        """Handle global overlay toggle."""
        self.model.global_overlay = checked
        self._update_all_views()

    def _on_alpha_changed(self, value: int) -> None:
        """Handle alpha slider changes."""
        alpha = value / 100.0
        self.model.global_alpha = alpha
        self.view.alpha_label.setText(f"Alpha: {alpha:.2f}")

        # Update all view configs
        for config in self.model.view_configs.values():
            config["alpha"] = alpha

        self._update_all_views()

    def _populate_label_color_controls(self) -> None:
        """Populate the label color controls based on current label data."""
        # Update the model's current label values
        self.model.update_current_label_values()
        label_values = self.model._current_label_values

        # Clear existing controls
        while self.view.label_colors_content_layout.count():
            child = self.view.label_colors_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.view.label_color_buttons.clear()

        if not label_values:
            # No labels - show empty message
            empty_label = QLabel(tr("no_labels_found"))
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: #666; font-style: italic;")
            self.view.label_colors_content_layout.addWidget(empty_label)
            self.view.reset_colors_btn.setEnabled(False)
            return

        # Create color controls for each label
        for label_val in label_values:
            # Create horizontal layout for this label
            label_layout = QHBoxLayout()

            # Label value text
            label_text = QLabel(f"{tr('label_prefix')} {label_val}:")
            label_text.setMinimumWidth(80)
            label_layout.addWidget(label_text)

            # Color button
            current_color = self.model.get_label_color(label_val)
            color_btn = ColorButton(color=current_color)
            color_btn.colorChanged.connect(
                partial(self._on_label_color_changed, label_val)
            )
            label_layout.addWidget(color_btn)

            # Add stretch to push everything left
            label_layout.addStretch()

            # Store button reference
            self.view.label_color_buttons[label_val] = color_btn

            # Create container widget and add to layout
            container = QWidget()
            container.setLayout(label_layout)
            self.view.label_colors_content_layout.addWidget(container)

        # Enable reset button
        self.view.reset_colors_btn.setEnabled(True)

        # Adjust scroll area height based on number of labels
        # Each label row: 24px button + 2px spacing + 2px margins = 28px total
        row_height = 28
        base_padding = 12  # Top and bottom padding in scroll area
        num_labels = len(label_values)

        if num_labels <= 10:
            # Show all labels without scroll for 10 or fewer
            optimal_height = num_labels * row_height + base_padding
            # Ensure minimum useful height
            optimal_height = max(optimal_height, 60)
        else:
            # Show exactly 10 labels, allow scrolling for more
            optimal_height = 10 * row_height + base_padding

        self.view.label_colors_scroll.setFixedHeight(optimal_height)

        # Make the group visible and expanded
        self.view.label_colors_group.setChecked(True)

    def _on_label_color_changed(self, label_value: int, color: QColor) -> None:
        """Handle label color changes from color buttons."""
        rgb_color = (color.red(), color.green(), color.blue())
        self.model.set_label_color(label_value, rgb_color)
        self._update_all_views()

    def _reset_label_colors(self) -> None:
        """Reset all label colors to defaults."""
        self.model.reset_label_colors()
        # Update button colors to show defaults
        for label_val, button in self.view.label_color_buttons.items():
            default_color = self.model.get_label_color(label_val)
            button.setColor(default_color)
        self._update_all_views()

    def _clear_label_color_controls(self) -> None:
        """Clear all label color controls."""
        # Clear existing controls
        while self.view.label_colors_content_layout.count():
            child = self.view.label_colors_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.view.label_color_buttons.clear()

        # Show empty message
        empty_label = QLabel(tr("no_labels_loaded"))
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("color: #666; font-style: italic;")
        self.view.label_colors_content_layout.addWidget(empty_label)

        # Disable reset button
        self.view.reset_colors_btn.setEnabled(False)

        # Reset scroll area to default height
        self.view.label_colors_scroll.setFixedHeight(80)

        # Collapse the group
        self.view.label_colors_group.setChecked(False)

    def _update_all_views(self) -> None:
        """Update all visible slice views."""
        for name, config in self.model.view_configs.items():
            if config["show"]:
                self._update_view(name)

    def _update_view(self, view_name: str) -> None:
        """Update a specific view."""
        if self.model.image_data is None:
            return

        config = self.model.view_configs[view_name]
        slice_idx = config["slice"]
        show_overlay = config["overlay"] and self.model.global_overlay
        alpha = config["alpha"]

        # Render slice
        qimage = self.model.render_slice(
            view_name, slice_idx, show_overlay, alpha)

        # Update view with rotation
        slice_view = self.view.slice_views[view_name]
        rotation = config["rotation"]
        slice_view.set_image(qimage, rotation)

        # Trigger preloading with debounce
        # 200ms delay to avoid excessive preloading
        self.preload_timer.start(200)

    def _preload_adjacent_slices(self) -> None:
        """Preload adjacent slices for all visible views."""
        if self.model.image_data is None:
            return

        for view_name, config in self.model.view_configs.items():
            if not config["show"]:
                continue

            current_slice = config["slice"]
            max_slice = self.model.get_max_slice(view_name)
            show_overlay = config["overlay"] and self.model.global_overlay
            alpha = config["alpha"]

            # Generate adjacent slice indices
            adjacent_slices = []
            for offset in range(1, self.preload_count + 1):
                # Forward slices
                if current_slice + offset <= max_slice:
                    adjacent_slices.append(current_slice + offset)
                # Backward slices
                if current_slice - offset >= 0:
                    adjacent_slices.append(current_slice - offset)

            if adjacent_slices:
                # Start preload worker
                worker = PreloadWorker(
                    self.model, view_name, adjacent_slices, show_overlay, alpha
                )
                self.preload_thread_pool.start(worker)


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main function to run the application."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create application
    app = QApplication(sys.argv)

    # Create MVC components
    model = ImageModel()
    controller = ViewerController(model)
    main_win = MainWindow(controller)
    controller.set_view(main_win)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="NIfTI/MHA Viewer")
    parser.add_argument("-i", "--image", help="Path to the input image file")
    parser.add_argument("-l", "--labels", help="Path to the input label file")
    parser.add_argument(
        "-o", "--output", help="Default output path for screenshots")
    args = parser.parse_args()

    # Load files from arguments
    if args.image:
        controller.thread_pool.start(
            LoadWorker(model, args.image, is_label=False))
    if args.labels:
        controller.thread_pool.start(
            LoadWorker(model, args.labels, is_label=True))
    if args.output:
        controller.default_output_path = args.output

    # Show main window and run application
    main_win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
