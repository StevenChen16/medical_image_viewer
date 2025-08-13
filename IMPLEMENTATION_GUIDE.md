# Quick Implementation Guide

This guide provides immediate actionable steps for implementing the next phase of development.

## Phase 1: Immediate Improvements (1-2 weeks)

### 1. Add Window/Level Adjustment

Add these methods to the `ViewerController` class:

```python
def setup_window_level_controls(self):
    """Add window/level controls to the control panel"""
    # Add to control dock widget
    wl_group = QGroupBox("Window/Level")
    wl_layout = QVBoxLayout(wl_group)
    
    # Preset dropdown
    self.wl_preset_combo = QComboBox()
    self.wl_preset_combo.addItems(['Brain', 'Bone', 'Lung', 'Abdomen', 'Custom'])
    
    # Manual controls
    self.window_spinbox = QSpinBox()
    self.window_spinbox.setRange(1, 5000)
    self.level_spinbox = QSpinBox() 
    self.level_spinbox.setRange(-1000, 3000)
    
    # Connect signals
    self.wl_preset_combo.currentTextChanged.connect(self.apply_preset_window_level)
    self.window_spinbox.valueChanged.connect(self.apply_manual_window_level)
    self.level_spinbox.valueChanged.connect(self.apply_manual_window_level)

def apply_window_level(self, window: int, level: int):
    """Apply window/level to image display"""
    if self.model.image_data is None:
        return
        
    # Calculate display range
    min_val = level - window // 2
    max_val = level + window // 2
    
    # Update all views
    for view_name in ['axial', 'sagittal', 'coronal']:
        self._update_view_window_level(view_name, min_val, max_val)
```

### 2. Implement Drag-and-Drop File Loading

Add to `MainWindow` class:

```python
def __init__(self):
    # ... existing code ...
    self.setAcceptDrops(True)

def dragEnterEvent(self, event):
    if event.mimeData().hasUrls():
        urls = event.mimeData().urls()
        if urls and any(url.toLocalFile().endswith(('.nii', '.nii.gz', '.mha', '.mhd')) 
                       for url in urls):
            event.accept()
        else:
            event.ignore()
    else:
        event.ignore()

def dropEvent(self, event):
    files = [url.toLocalFile() for url in event.mimeData().urls()]
    if files:
        # Load first file as image, second as labels if available
        self.controller.load_image(files[0])
        if len(files) > 1:
            self.controller.load_labels(files[1])
```

### 3. Add Area Measurement Tool

Extend `MeasurementManager` class:

```python
@dataclass
class AreaMeasurement:
    id: int
    type: str = "area"
    view_name: str = ""
    slice_idx: int = 0
    points: List[Tuple[float, float]] = field(default_factory=list)
    area_px: float = 0.0
    area_mm2: float = 0.0
    closed: bool = False
    timestamp: float = 0.0

class MeasurementManager(QObject):
    def start_area_measurement(self, view_name: str, slice_idx: int, start_pos: tuple):
        """Start drawing area measurement"""
        measurement = AreaMeasurement(
            id=self._next_id,
            view_name=view_name,
            slice_idx=slice_idx,
            points=[start_pos],
            timestamp=time.time()
        )
        self.measurements[self._next_id] = measurement
        self._next_id += 1
        return measurement
    
    def add_area_point(self, measurement_id: int, point: tuple):
        """Add point to area measurement"""
        if measurement_id in self.measurements:
            measurement = self.measurements[measurement_id]
            measurement.points.append(point)
            self._calculate_area(measurement)
    
    def close_area_measurement(self, measurement_id: int):
        """Complete area measurement"""
        if measurement_id in self.measurements:
            measurement = self.measurements[measurement_id]
            measurement.closed = True
            self._calculate_area(measurement)
            self.measurementAdded.emit(measurement)
```

## Phase 2: Code Organization (1-2 weeks)

### Create Module Structure

1. Create the directory structure as outlined in TECHNICAL_ANALYSIS.md
2. Move classes to appropriate modules:

```bash
mkdir -p medical_image_viewer/{models,views,controllers,widgets,utils,workers}
# Create __init__.py files
touch medical_image_viewer/__init__.py
touch medical_image_viewer/{models,views,controllers,widgets,utils,workers}/__init__.py
```

3. Split nifti_viewer.py:
   - Move `ImageModel` → `models/image_model.py`
   - Move `Measurement*` classes → `models/measurement_model.py`
   - Move `MainWindow` → `views/main_window.py`
   - Move `SliceView` → `views/slice_view.py`
   - Move `ViewerController` → `controllers/viewer_controller.py`

### Update Imports

Update main.py entry point:

```python
#!/usr/bin/env python3
"""
Medical Image Viewer - Entry Point
"""
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from medical_image_viewer.views.main_window import MainWindow
from medical_image_viewer.controllers.viewer_controller import ViewerController
# ... other imports

def main():
    # Existing main function code
    pass

if __name__ == "__main__":
    main()
```

## Phase 3: Testing Framework (1 week)

### Set up pytest

1. Install testing dependencies:
```bash
pip install pytest pytest-qt pytest-cov
```

2. Create test files:
```python
# tests/conftest.py
import pytest
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()

# tests/test_image_model.py
import pytest
import numpy as np
from medical_image_viewer.models.image_model import ImageModel

class TestImageModel:
    def test_initialization(self):
        model = ImageModel()
        assert model.image_data is None
        assert model.label_data is None
    
    def test_slice_navigation(self):
        model = ImageModel()
        # Create test data
        test_data = np.random.rand(100, 100, 50)
        model.image_data = test_data
        model.image_shape = test_data.shape
        
        # Test axial slicing
        assert model.get_max_slice('axial') == 49
        slice_data = model.get_slice('axial', 25)
        assert slice_data.shape == (100, 100)
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=medical_image_viewer tests/

# Run specific test
pytest tests/test_image_model.py::TestImageModel::test_initialization
```

## Quick Wins for User Experience

### 1. Keyboard Shortcuts

Add to `MainWindow.setup_shortcuts()`:

```python
# File operations
QShortcut(QKeySequence("Ctrl+O"), self, self.open_image_file)
QShortcut(QKeySequence("Ctrl+L"), self, self.open_labels_file)
QShortcut(QKeySequence("Ctrl+R"), self, self.reset_views)

# View controls
QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
QShortcut(QKeySequence("Space"), self, self.toggle_overlay)

# Measurement tools
QShortcut(QKeySequence("M"), self, self.start_line_measurement)
QShortcut(QKeySequence("A"), self, self.start_area_measurement)
QShortcut(QKeySequence("Delete"), self, self.delete_selected_measurement)
```

### 2. Recent Files Menu

Add to `MainWindow`:

```python
def update_recent_files(self, filepath: str):
    """Update recent files list"""
    # Load from settings
    recent_files = self.settings.value("recent_files", [])
    if filepath in recent_files:
        recent_files.remove(filepath)
    recent_files.insert(0, filepath)
    recent_files = recent_files[:10]  # Keep only 10 recent files
    
    # Save to settings
    self.settings.setValue("recent_files", recent_files)
    self.update_recent_files_menu()

def update_recent_files_menu(self):
    """Update recent files menu"""
    self.recent_files_menu.clear()
    recent_files = self.settings.value("recent_files", [])
    for filepath in recent_files:
        if Path(filepath).exists():
            action = self.recent_files_menu.addAction(Path(filepath).name)
            action.setData(filepath)
            action.triggered.connect(
                lambda checked, path=filepath: self.controller.load_image(path)
            )
```

## Testing Your Changes

1. **Backup current working version**: 
   ```bash
   cp nifti_viewer.py nifti_viewer_backup.py
   ```

2. **Test each feature incrementally**:
   - Implement one feature at a time
   - Test with sample data after each change
   - Verify existing functionality still works

3. **Sample test data**: Create small test files for development:
   ```python
   # create_test_data.py
   import nibabel as nib
   import numpy as np
   
   # Create test NIfTI file
   data = np.random.rand(64, 64, 32) * 1000
   img = nib.Nifti1Image(data, np.eye(4))
   nib.save(img, 'test_image.nii.gz')
   
   # Create test labels
   labels = np.random.randint(0, 5, (64, 64, 32))
   label_img = nib.Nifti1Image(labels, np.eye(4))
   nib.save(label_img, 'test_labels.nii.gz')
   ```

This guide provides concrete, implementable steps to begin the next phase of development while maintaining the stability of the existing codebase.