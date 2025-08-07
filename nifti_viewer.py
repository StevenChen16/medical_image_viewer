#!/usr/bin/env python3
"""
NIfTI/MHA Viewer with PySide 6
A medical image viewer for NIfTI and MHA format files.
"""

__version__ = "0.1.3"
__author__ = "Steven Chen"
__license__ = "MIT"
__copyright__ = "Copyright 2025, Steven Chen"
__all__ = ['MainWindow', 'ViewerController', 'ImageModel', 'main']

import sys
import os
import logging
import argparse
from pathlib import Path
from functools import lru_cache, partial
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
    QDialog, QTabWidget, QTextEdit, QScrollArea, QSizePolicy, QColorDialog
)
from PySide6.QtCore import (
    Qt, QObject, Signal, QThread, QRunnable, QThreadPool, QTimer,
    QRect, QSize, QPointF
)
from PySide6.QtGui import (
    QPixmap, QPainter, QImage, QPen, QBrush, QColor, QTransform,
    QAction, QIcon, QFont, QWheelEvent, QMouseEvent, QPixmapCache,
    QShortcut, QCursor
)


# ============================================================================
# UTILITY WIDGETS
# ============================================================================

class ColorButton(QPushButton):
    """Custom color picker button widget."""
    
    colorChanged = Signal(object)  # Emits QColor when color changes
    
    def __init__(self, *args, color=None, **kwargs):
        super().__init__(*args, **kwargs)
        if color is None:
            self._color = QColor(227, 26, 28)  # Default to first enhanced color (red)
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
            self.setStyleSheet(f"""
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
            """)
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
    imageLoaded = Signal(str, tuple)  # filename, shape
    labelLoaded = Signal(str, int)    # filename, unique_labels_count
    sliceReady = Signal(str, int, QImage)  # view_name, slice_idx, image
    loadError = Signal(str)           # error_message
    loadProgress = Signal(int)        # loading progress percentage
    
    def __init__(self, pixmap_cache_size: int = 102400):  # 100MB default
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

        # Medical-optimized ColorBrewer Paired palette - reordered for maximum contrast
        # Alternates between different hue families for better medical visualization
        # Original ColorBrewer colors maintained but resequenced for consecutive contrast
        self._default_colors = [
            (227,  26,  28), # 1. Red - high contrast, medically significant
            (178, 223, 138), # 2. Light green - strong contrast to red
            ( 31, 120, 180), # 3. Dark blue - distinct from green
            (253, 191, 111), # 4. Orange - warm contrast to blue
            (106,  61, 154), # 5. Purple - cool contrast to orange
            (255, 255, 153), # 6. Yellow - bright contrast to purple
            (166, 206, 227), # 7. Light blue - different from dark blue above
            (177,  89,  40), # 8. Brown - earth tone contrast
            (251, 154, 153), # 9. Light red - softer than primary red
            ( 51, 160,  44), # 10. Dark green - different from light green
            (255, 127,   0), # 11. Dark orange - vibrant accent
            (202, 178, 214), # 12. Light purple - subtle final color
        ]
        
        # Dynamic color mapping: label_value -> (r, g, b) tuple
        self._custom_label_colors = {}  # User customizations
        self._current_label_values = []  # Currently loaded label values
        
        # Fixed overlay transparency for clinical consistency
        self._overlay_alpha = 0.4
        
        # Configure QPixmapCache
        QPixmapCache.setCacheLimit(pixmap_cache_size)
        
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
            p = p.with_suffix('')
        return str(p)

    def _normalize_output_path(self, out_path: str) -> Tuple[str, str]:
        """Ensure path has a single valid extension and return (path, ext)."""
        p = Path(out_path)
        suffixes = p.suffixes
        if suffixes[-2:] == ['.nii', '.gz']:
            ext = '.nii.gz'
        elif suffixes:
            ext = suffixes[-1].lower()
        else:
            raise ValueError("Output path must have an extension")

        base = Path(self.strip_extensions(out_path))
        return str(base.with_suffix(ext)), ext

    def save_volume(self, data: np.ndarray, reference_path: str, out_path: str) -> str:
        """Save a 3D volume to disk using reference metadata."""
        out_path, ext = self._normalize_output_path(out_path)
        if ext in ['.nii', '.nii.gz']:
            ref_img = nib.load(reference_path)
            img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
            nib.save(img, out_path)
        elif ext in ['.mha', '.mhd']:
            if not SITK_AVAILABLE:
                raise RuntimeError("SimpleITK is required to save MHA/MHD files")
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
                ovr_gray = (0.299 * color_ovr[:, :, 0] +
                            0.587 * color_ovr[:, :, 1] +
                            0.114 * color_ovr[:, :, 2]).astype(np.uint8)
                alpha = self._overlay_alpha
                blended = ((1 - alpha) * slice_img + alpha * ovr_gray).astype(np.uint8)
            else:
                blended = slice_img
            overlay_vol[:, :, k] = blended

        return overlay_vol

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
        has_overlay = (show_overlay and self.label_data is not None and 
                      self.get_slice_data(axis, slice_idx, True).size > 0)
        
        if has_overlay:
            label_slice = self.get_slice_data(axis, slice_idx, True)
            overlay = self.create_label_overlay(label_slice)
            if overlay is not None:
                # Medical-grade alpha blending with ColorBrewer Paired colors
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
                mask = label_slice > 0
                
                # Use clinical-grade alpha with user adjustment
                # Base clinical alpha (0.4) modulated by user alpha setting
                clinical_alpha = self._overlay_alpha * alpha  # Balanced clinical visibility
                img_rgb = img_rgb.astype(np.float32, copy=False)
                overlay_f = overlay.astype(np.float32, copy=False)
                
                # Alpha blend: base * (1-alpha) + overlay * alpha
                img_rgb[mask] = ((1 - clinical_alpha) * img_rgb[mask] + 
                               clinical_alpha * overlay_f[mask])
                
                img_rgb = img_rgb.astype(np.uint8, copy=False)
                img_rgb = np.flipud(img_rgb)
                img_rgb = np.ascontiguousarray(img_rgb)
                height, width, _ = img_rgb.shape
                bytes_per_line = 3 * width
                qimage = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                has_overlay = False
        
        if not has_overlay:
            # Use efficient grayscale format
            img_flipped = np.flipud(img_normalized)
            img_flipped = np.ascontiguousarray(img_flipped)
            height, width = img_flipped.shape
            qimage = QImage(img_flipped.data, width, height, img_flipped.strides[0], QImage.Format_Grayscale8)

        if rotation != 0 and not qimage.isNull():
            transform = QTransform()
            transform.rotate(rotation)
            qimage = qimage.transformed(transform)

        return qimage
    
    def load_image(self, filepath: str) -> None:
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

    def load_labels(self, filepath: str) -> None:
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
    
    def set_slice(self, view_name: str, slice_idx: int) -> None:
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
    
    def get_label_color(self, label_value: int) -> Tuple[int, int, int]:
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
    
    def _enhance_color_for_medical_display(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Enhance color contrast for medical imaging dark backgrounds.
        Boosts brightness and saturation while preserving hue relationships.
        """
        r, g, b = color
        
        # Convert to HSV for easier manipulation
        r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
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
    
    def set_label_color(self, label_value: int, color: Tuple[int, int, int]) -> None:
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
    
    def __init__(self, model: ImageModel, view_name: str, slice_indices: list, 
                 show_overlay: bool = True, alpha: float = 0.5):
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
                    self.model.render_slice(self.view_name, slice_idx, 
                                          self.show_overlay, self.alpha)
        except Exception as e:
            # Silent fail for preloading to avoid disrupting user experience
            pass


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
    
    def create_english_tab(self) -> None:
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
        version_label = QLabel("Version 0.1.3")
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
            <li><b>Save:</b> File → Save Screenshot (Ctrl+S) or Volume (Ctrl+Shift+S)</li>
        </ul>
        
        <h3 style=\"color: #4a90e2;\">Keyboard Shortcuts</h3>
        <table style=\"width: 100%; border-collapse: collapse;">
            <tr><td style=\"padding: 4px;\"><b>Ctrl+O</b></td><td style=\"padding: 4px;\">Open image file</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+L</b></td><td style=\"padding: 4px;\">Open label file</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+S</b></td><td style=\"padding: 4px;\">Save screenshot</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+Shift+S</b></td><td style=\"padding: 4px;\">Open volume save menu</td></tr>
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
    
    def create_chinese_tab(self) -> None:
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
        version_label = QLabel("版本 0.1.3")
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
            <li><b>保存：</b>文件 → 保存截图 (Ctrl+S) 或保存体数据 (Ctrl+Shift+S)</li>
        </ul>
        
        <h3 style=\"color: #4a90e2;\">快捷键一览</h3>
        <table style=\"width: 100%; border-collapse: collapse;">
            <tr><td style=\"padding: 4px;\"><b>Ctrl+O</b></td><td style=\"padding: 4px;\">打开影像文件</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+L</b></td><td style=\"padding: 4px;\">打开标签文件</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+S</b></td><td style=\"padding: 4px;\">保存截图</td></tr>
            <tr><td style=\"padding: 4px;\"><b>Ctrl+Shift+S</b></td><td style=\"padding: 4px;\">打开体数据保存菜单</td></tr>
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
    
    def set_model(self, model: ImageModel) -> None:
        """Connect to data model."""
        self.model = model
    
    def set_image(self, qimage: QImage) -> None:
        """Update displayed image with QPixmapCache optimization."""
        if qimage.isNull():
            self.pixmap_item.setPixmap(QPixmap())
            return
        
        # Generate cache key based on image properties and view name
        cache_key = f"{self.view_name}_{qimage.cacheKey()}_{qimage.format()}"
        pixmap = QPixmapCache.find(cache_key)
        
        if pixmap is None:
            pixmap = QPixmap.fromImage(qimage)
            # Insert with size-aware caching
            QPixmapCache.insert(cache_key, pixmap)
        
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
    
    def reset_view(self) -> None:
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
    
    def setup_control_dock(self) -> None:
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

        # Label Colors controls
        self.label_colors_layout = QVBoxLayout()
        self.label_colors_scroll = QScrollArea()
        self.label_colors_scroll.setWidgetResizable(True)
        self.label_colors_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.label_colors_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Initial fixed height - will be dynamically adjusted when labels are loaded
        self.label_colors_scroll.setFixedHeight(80)
        
        self.label_colors_widget = QWidget()
        self.label_colors_content_layout = QVBoxLayout(self.label_colors_widget)
        self.label_colors_content_layout.setContentsMargins(4, 4, 4, 4)
        self.label_colors_content_layout.setSpacing(2)  # Tight spacing between label rows
        self.label_colors_scroll.setWidget(self.label_colors_widget)
        
        # Add initial empty label inside the scroll area
        empty_label = QLabel("No labels loaded")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("color: #666; font-style: italic;")
        self.label_colors_content_layout.addWidget(empty_label)
        
        self.label_colors_layout.addWidget(self.label_colors_scroll)
        
        # Reset colors button
        self.reset_colors_btn = QPushButton("Reset to Defaults")
        self.reset_colors_btn.setEnabled(False)
        self.label_colors_layout.addWidget(self.reset_colors_btn)
        
        self.label_colors_group = create_collapsible_group("Label Colors", self.label_colors_layout)
        self.label_colors_group.setChecked(False)  # Start collapsed
        dock_layout.addWidget(self.label_colors_group)
        
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
    
    def setup_menu(self) -> None:
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

        save_submenu = file_menu.addMenu("Save")
        save_image_action = QAction("Image Only...", self)
        save_label_action = QAction("Label Only...", self)
        save_overlay_action = QAction("Overlay Image...", self)
        save_submenu.addAction(save_image_action)
        save_submenu.addAction(save_label_action)
        save_submenu.addAction(save_overlay_action)

        # Shortcut to quickly open the Save submenu
        self.save_shortcut = QShortcut("Ctrl+Shift+S", self)
        self.save_shortcut.activated.connect(lambda: save_submenu.exec(QCursor.pos()))

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
            'save_image': save_image_action,
            'save_label': save_label_action,
            'save_overlay': save_overlay_action,
            'save_screenshot': save_screenshot_action,
            'reset': reset_action,
            'exit': exit_action,
            'fit_all': fit_all_action,
            'toggle_control_panel': toggle_control_panel_action,
            'about': about_action
        }
    
    def setup_statusbar(self) -> None:
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
    
    def __init__(self, model: ImageModel, view: MainWindow, preload_count: int = 2):
        super().__init__()
        
        self.model = model
        self.view = view
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
        
        # Setup connections
        self.setup_model_connections()
        self.setup_view_connections()
        self.setup_slice_view_connections()
        
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
        self.view.actions['load_image'].triggered.connect(self.load_image)
        self.view.actions['load_labels'].triggered.connect(self.load_labels)
        self.view.actions['reset'].triggered.connect(self.reset_all)
        self.view.actions['exit'].triggered.connect(self.view.close)
        self.view.actions['fit_all'].triggered.connect(self.fit_all_views)
        self.view.actions['save_image'].triggered.connect(self._save_image)
        self.view.actions['save_label'].triggered.connect(self._save_label)
        self.view.actions['save_overlay'].triggered.connect(self._save_overlay)
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
            checkbox.toggled.connect(partial(self._toggle_view_visibility, name))
        
        # Slice controls
        for name, controls in self.view.slice_controls.items():
            slider = controls['slider']
            spinbox = controls['spinbox']
            rotate_btn = controls['rotate_btn']
            
            slider.valueChanged.connect(partial(self._on_slice_changed, name))
            spinbox.valueChanged.connect(partial(self._on_slice_changed, name))
            rotate_btn.clicked.connect(partial(self._rotate_view, name))
        
        # Overlay controls
        self.view.global_overlay_cb.toggled.connect(self._on_global_overlay_toggled)
        self.view.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        
        # Label color controls
        self.view.reset_colors_btn.clicked.connect(self._reset_label_colors)
        
        # Control panel toggle button
        self.view.panel_toggle_btn.clicked.connect(self.toggle_control_panel)
    
    def setup_slice_view_connections(self) -> None:
        """Connect slice view signals."""
        for name, slice_view in self.view.slice_views.items():
            slice_view.wheelScrolled.connect(partial(self._on_wheel_scroll, name))
            slice_view.mousePositionChanged.connect(partial(self._on_mouse_position, name))
    
    def load_image(self) -> None:
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
    
    def load_labels(self) -> None:
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
    
    def update_image_from_path(self) -> None:
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
    
    def update_labels_from_path(self) -> None:
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
    
    def reset_all(self) -> None:
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
            
            # Clear label color controls
            self._clear_label_color_controls()
        
        for slice_view in self.view.slice_views.values():
            slice_view.scene.clear()
            slice_view.pixmap_item = QGraphicsPixmapItem()
            slice_view.scene.addItem(slice_view.pixmap_item)
        
        self.view.status_label.setText("Ready - Load an image to begin")
        
        # Reset progress bar if control panel exists
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
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
        default = self.model.strip_extensions(self.model.image_path) if self.model.image_path else ""
        path, _ = QFileDialog.getSaveFileName(self.view, "Save Image Only", default, filters)
        if path:
            try:
                path = self.model.save_volume(self.model.image_data, self.model.image_path, path)
                self.view.status_label.setText(f"Image saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))

    def _save_label(self) -> None:
        if self.model.label_data is None:
            QMessageBox.warning(self.view, "Warning", "No label loaded.")
            return
        filters = "NIfTI (*.nii *.nii.gz);;MetaImage (*.mha *.mhd);;All Files (*)"
        default = self.model.strip_extensions(self.model.label_path) if self.model.label_path else ""
        path, _ = QFileDialog.getSaveFileName(self.view, "Save Label Only", default, filters)
        if path:
            try:
                path = self.model.save_volume(self.model.label_data, self.model.label_path, path)
                self.view.status_label.setText(f"Label saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))

    def _save_overlay(self) -> None:
        if self.model.image_data is None or self.model.label_data is None:
            QMessageBox.warning(self.view, "Warning", "Need both image and label loaded.")
            return
        filters = "NIfTI (*.nii *.nii.gz);;MetaImage (*.mha *.mhd);;All Files (*)"
        default = self.model.strip_extensions(self.model.image_path) if self.model.image_path else ""
        path, _ = QFileDialog.getSaveFileName(self.view, "Save Overlay Image", default, filters)
        if path:
            try:
                overlay_vol = self.model.make_overlay_volume()
                path = self.model.save_volume(overlay_vol, self.model.image_path, path)
                self.view.status_label.setText(f"Overlay saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))

    def save_screenshot(self) -> None:
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
    
    def show_about(self) -> None:
        """Show About dialog."""
        about_dialog = AboutDialog(self.view)
        about_dialog.exec()
    
    def toggle_control_panel(self) -> None:
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
    
    def _reconnect_control_signals(self) -> None:
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
            checkbox.toggled.connect(partial(self._toggle_view_visibility, name))
        
        # Slice controls
        for name, controls in self.view.slice_controls.items():
            slider = controls['slider']
            spinbox = controls['spinbox']
            rotate_btn = controls['rotate_btn']
            
            slider.valueChanged.connect(partial(self._on_slice_changed, name))
            spinbox.valueChanged.connect(partial(self._on_slice_changed, name))
            rotate_btn.clicked.connect(partial(self._rotate_view, name))
        
        # Overlay controls
        self.view.global_overlay_cb.toggled.connect(self._on_global_overlay_toggled)
        self.view.alpha_slider.valueChanged.connect(self._on_alpha_changed)
    
    def _on_image_loaded(self, filepath: str, shape: tuple) -> None:
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
    
    def _on_labels_loaded(self, filepath: str, unique_count: int) -> None:
        """Handle successful label loading."""
        self.view.status_label.setText(f"Labels loaded: {Path(filepath).name} - {unique_count} unique labels")
        
        # Update control panel elements if they exist
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)
            # Update label path input
            self.view.label_path_input.setText(filepath)
            # Populate color controls
            self._populate_label_color_controls()
        
        self._update_all_views()
    
    def _on_load_error(self, error_msg: str) -> None:
        """Handle loading errors."""
        self.view.status_label.setText("Load failed")
        
        # Hide progress bar if control panel exists
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setVisible(False)

        QMessageBox.critical(self.view, "Load Error", error_msg)

    def _on_load_progress(self, value: int) -> None:
        """Update progress bar during loading."""
        if hasattr(self.view, 'control_dock') and self.view.control_dock is not None:
            self.view.progress_bar.setValue(value)
            if value >= 100:
                self.view.progress_bar.setVisible(False)
    
    def _toggle_view_visibility(self, view_name: str, visible: bool) -> None:
        """Toggle view visibility."""
        self.model.view_configs[view_name]['show'] = visible
        
        # Find the view frame and hide/show it
        view_index = ['axial', 'sagittal', 'coronal'].index(view_name)
        widget = self.view.splitter.widget(view_index)
        widget.setVisible(visible)
    
    def _on_slice_changed(self, view_name: str, value: int) -> None:
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
    
    def _on_wheel_scroll(self, view_name: str, delta: int) -> None:
        """Handle mouse wheel slice navigation."""
        controls = self.view.slice_controls[view_name]
        current_value = controls['slider'].value()
        new_value = current_value + delta
        
        # Clamp to valid range
        max_value = controls['slider'].maximum()
        new_value = max(0, min(new_value, max_value))
        
        self._on_slice_changed(view_name, new_value)
    
    def _on_mouse_position(self, view_name: str, x: int, y: int) -> None:
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
    
    def _rotate_view(self, view_name: str) -> None:
        """Rotate view by 90 degrees."""
        config = self.model.view_configs[view_name]
        config['rotation'] = (config['rotation'] + 90) % 360
        self._update_view(view_name)
    
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
            config['alpha'] = alpha
        
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
            empty_label = QLabel("No labels found")
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
            label_text = QLabel(f"Label {label_val}:")
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
        empty_label = QLabel("No labels loaded")
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
            if config['show']:
                self._update_view(name)
    
    def _update_view(self, view_name: str) -> None:
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
        
        # Trigger preloading with debounce
        self.preload_timer.start(200)  # 200ms delay to avoid excessive preloading
    
    def _preload_adjacent_slices(self) -> None:
        """Preload adjacent slices for all visible views."""
        if self.model.image_data is None:
            return
        
        for view_name, config in self.model.view_configs.items():
            if not config['show']:
                continue
                
            current_slice = config['slice']
            max_slice = self.model.get_max_slice(view_name)
            show_overlay = config['overlay'] and self.model.global_overlay
            alpha = config['alpha']
            
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
                worker = PreloadWorker(self.model, view_name, adjacent_slices, 
                                     show_overlay, alpha)
                self.preload_thread_pool.start(worker)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
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
    parser.add_argument("--pixmap-cache-size", type=int, default=102400, 
                        help="QPixmapCache size in KB (default: 100MB)")
    parser.add_argument("--preload-count", type=int, default=2, 
                        help="Number of adjacent slices to preload (default: 2)")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s 0.1.3")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help="Set logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Medical Image Viewer")
    app.setApplicationVersion("0.1.3")
    
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
        model = ImageModel(pixmap_cache_size=args.pixmap_cache_size)
        view = MainWindow()
        controller = ViewerController(model, view, preload_count=args.preload_count)
        
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
