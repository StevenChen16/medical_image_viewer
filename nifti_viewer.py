#!/usr/bin/env python3
"""
NIfTI Viewer with Label Overlay
A simple tkinter-based viewer for NIfTI files with label overlay capability.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import nibabel as nib
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


class NIfTIViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("NIfTI Viewer with Label Overlay")
        self.root.geometry("1200x800")
        
        # Data storage
        self.image_data = None
        self.label_data = None
        self.show_overlay = True  # Global overlay switch
        self.overlay_alpha = 0.5
        
        # File paths
        self.image_path = ""
        self.label_path = ""
        
        # Three-view data
        self.views = {
            'sagittal': {'axis': 0, 'slice': 0, 'show': True, 'overlay': True, 'rotation': 0, 'alpha': 0.5},
            'coronal': {'axis': 1, 'slice': 0, 'show': True, 'overlay': True, 'rotation': 0, 'alpha': 0.5},
            'axial': {'axis': 2, 'slice': 0, 'show': True, 'overlay': True, 'rotation': 0, 'alpha': 0.5}
        }
        
        # Original view proportions
        self.view_weights = {
            'axial': 6,
            'sagittal': 2, 
            'coronal': 1
        }
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File loading buttons
        ttk.Button(control_frame, text="Load Image", 
                  command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Load Labels", 
                  command=self.load_labels).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Reset", 
                  command=self.reset_viewer).pack(side=tk.LEFT, padx=(0, 20))
        
        # View visibility controls
        ttk.Label(control_frame, text="Views:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.view_visibility_vars = {}
        view_names = ['axial', 'sagittal', 'coronal']
        view_display_names = ['Axial', 'Sagittal', 'Coronal']
        
        for view_name, display_name in zip(view_names, view_display_names):
            var = tk.BooleanVar(value=True)
            self.view_visibility_vars[view_name] = var
            ttk.Checkbutton(control_frame, text=display_name, 
                           variable=var,
                           command=self.update_view_layout).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Separator(control_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=(10, 10))
        
        # Global overlay controls
        ttk.Label(control_frame, text="Global:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Overlay controls
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Overlay", 
                       variable=self.overlay_var,
                       command=self.toggle_overlay).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(control_frame, text="Alpha:").pack(side=tk.LEFT, padx=(0, 5))
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, 
                               variable=self.alpha_var, length=100,
                               command=self.change_alpha)
        alpha_scale.pack(side=tk.LEFT, padx=(0, 20))
        
        # File path input frame
        path_frame = ttk.Frame(main_frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Image path input
        ttk.Label(path_frame, text="Image Path:").pack(side=tk.LEFT, padx=(0, 5))
        self.image_path_var = tk.StringVar()
        self.image_path_entry = ttk.Entry(path_frame, textvariable=self.image_path_var, width=40)
        self.image_path_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(path_frame, text="Update Image", 
                  command=self.update_image_from_path).pack(side=tk.LEFT, padx=(0, 20))
        
        # Label path input
        ttk.Label(path_frame, text="Label Path:").pack(side=tk.LEFT, padx=(0, 5))
        self.label_path_var = tk.StringVar()
        self.label_path_entry = ttk.Entry(path_frame, textvariable=self.label_path_var, width=40)
        self.label_path_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(path_frame, text="Update Labels", 
                  command=self.update_labels_from_path).pack(side=tk.LEFT, padx=(0, 20))
        
        # Three-view display frame
        views_frame = ttk.Frame(main_frame)
        views_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create three view panels
        self.view_frames = {}
        self.canvases = {}
        self.view_vars = {}
        self.overlay_vars = {}
        self.alpha_vars = {}  # Individual alpha variables for each view
        self.alpha_scales = {}  # Individual alpha scales for each view
        self.slice_vars = {}
        self.slice_scales = {}
        self.slice_labels = {}
        
        view_names = ['axial', 'sagittal', 'coronal']
        view_titles = ['Axial (XY)', 'Sagittal (YZ)', 'Coronal (XZ)']
        view_weights = [6, 2, 1]  # Width proportions: Axial=6, Sagittal=2, Coronal=1
        
        for i, (view_name, view_title) in enumerate(zip(view_names, view_titles)):
            # Create frame for each view
            view_frame = ttk.LabelFrame(views_frame, text=view_title, padding=5)
            view_frame.grid(row=0, column=i, sticky="nsew", padx=5, pady=5)
            self.view_frames[view_name] = view_frame
            
            # Canvas for image display
            canvas = tk.Canvas(view_frame, bg="black", height=520, width=300)
            canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.canvases[view_name] = canvas
            
            # Slice navigation for this view
            slice_frame = ttk.Frame(view_frame)
            slice_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(slice_frame, text="Slice:").pack(side=tk.LEFT, padx=(0, 2))
            slice_var = tk.IntVar()
            self.slice_vars[view_name] = slice_var
            
            slice_scale = ttk.Scale(slice_frame, from_=0, to=100, 
                                  variable=slice_var, length=150,
                                  command=lambda val, v=view_name: self.change_slice(val, v))
            slice_scale.pack(side=tk.LEFT, padx=(0, 5))
            self.slice_scales[view_name] = slice_scale
            
            slice_label = ttk.Label(slice_frame, text="0/0")
            slice_label.pack(side=tk.LEFT)
            self.slice_labels[view_name] = slice_label
            
            # Control checkboxes for this view
            control_frame = ttk.Frame(view_frame)
            control_frame.pack(fill=tk.X)
            
            # Create two rows for controls
            controls_top = ttk.Frame(control_frame)
            controls_top.pack(fill=tk.X)
            controls_bottom = ttk.Frame(control_frame)
            controls_bottom.pack(fill=tk.X, pady=(5, 0))
            
            # Top row: Show overlay checkbox (centered)
            overlay_var = tk.BooleanVar(value=True)
            self.overlay_vars[view_name] = overlay_var
            overlay_checkbox = ttk.Checkbutton(controls_top, text="Show Overlay", 
                                             variable=overlay_var,
                                             command=lambda v=view_name: self.toggle_view_overlay(v))
            overlay_checkbox.pack()
            
            # Bottom row: Alpha control and rotation button
            ttk.Label(controls_bottom, text="α:").pack(side=tk.LEFT, padx=(0, 2))
            alpha_var = tk.DoubleVar(value=0.5)
            self.alpha_vars[view_name] = alpha_var
            alpha_scale = ttk.Scale(controls_bottom, from_=0.0, to=1.0, 
                                  variable=alpha_var, length=80,
                                  command=lambda val, v=view_name: self.change_view_alpha(val, v))
            alpha_scale.pack(side=tk.LEFT, padx=(0, 5))
            self.alpha_scales[view_name] = alpha_scale
            
            # Rotation button
            ttk.Button(controls_bottom, text="↻ Rotate", width=8,
                      command=lambda v=view_name: self.rotate_view(v)).pack(side=tk.RIGHT)
        
        # Configure grid weights - will be updated dynamically
        self.update_view_layout()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image and labels to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
    def load_image(self):
        """Load the main NIfTI image"""
        filename = filedialog.askopenfilename(
            title="Select NIfTI Image File",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Reset all data when loading a new image
                self.reset_data()
                
                img = nib.load(filename)
                self.image_data = img.get_fdata()
                self.image_path = filename
                self.image_path_var.set(filename)
                self.update_slice_range()
                self.status_var.set(f"Image loaded: {Path(filename).name} - Shape: {self.image_data.shape}")
                self.update_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def load_labels(self):
        """Load the label NIfTI file"""
        filename = filedialog.askopenfilename(
            title="Select NIfTI Label File",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                label_img = nib.load(filename)
                new_label_data = label_img.get_fdata()
                
                # Check if image data exists and if shapes match
                if self.image_data is not None:
                    if new_label_data.shape != self.image_data.shape:
                        messagebox.showwarning("Warning", 
                                             f"Label shape {new_label_data.shape} doesn't match image shape {self.image_data.shape}")
                        return  # Don't load mismatched labels
                
                self.label_data = new_label_data
                self.label_path = filename
                self.label_path_var.set(filename)
                unique_labels = np.unique(self.label_data)
                self.status_var.set(f"Labels loaded: {Path(filename).name} - {len(unique_labels)} unique labels")
                self.update_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load labels: {str(e)}")
                
    def update_slice_range(self):
        """Update the slice slider ranges for all views"""
        if self.image_data is None:
            return
            
        for view_name, view_data in self.views.items():
            axis = view_data['axis']
            max_slice = self.image_data.shape[axis] - 1
            self.slice_scales[view_name].config(to=max_slice)
            self.views[view_name]['slice'] = min(self.views[view_name]['slice'], max_slice)
            self.slice_vars[view_name].set(self.views[view_name]['slice'])
            self.slice_labels[view_name].config(text=f"{self.views[view_name]['slice']}/{max_slice}")
        
    def change_slice(self, value, view_name):
        """Change slice for a specific view"""
        slice_num = int(float(value))
        self.views[view_name]['slice'] = slice_num
        
        if self.image_data is not None:
            axis = self.views[view_name]['axis']
            max_slice = self.image_data.shape[axis] - 1
            self.slice_labels[view_name].config(text=f"{slice_num}/{max_slice}")
        
        self.update_view_display(view_name)
        
    def update_view_layout(self):
        """Update the layout of visible views"""
        # Get list of visible views in display order
        visible_views = []
        for view_name in ['axial', 'sagittal', 'coronal']:
            if self.view_visibility_vars[view_name].get():
                visible_views.append(view_name)
                self.views[view_name]['show'] = True
                self.view_frames[view_name].grid()
            else:
                self.views[view_name]['show'] = False
                self.view_frames[view_name].grid_remove()
        
        # Rearrange visible views
        views_frame = self.view_frames['axial'].master
        
        # Clear all column weights
        for i in range(3):
            views_frame.columnconfigure(i, weight=0)
        
        # Redistribute visible views with proper proportions
        for i, view_name in enumerate(visible_views):
            self.view_frames[view_name].grid(row=0, column=i, sticky="nsew", padx=5, pady=5)
            # Use original proportional weights
            views_frame.columnconfigure(i, weight=self.view_weights[view_name])
        
        # Update display for all views
        self.update_display()
        
    def toggle_view_overlay(self, view_name):
        """Toggle overlay for a specific view"""
        self.views[view_name]['overlay'] = self.overlay_vars[view_name].get()
        self.update_view_display(view_name)
        
    def rotate_view(self, view_name):
        """Rotate a specific view by 90 degrees"""
        self.views[view_name]['rotation'] = (self.views[view_name]['rotation'] + 90) % 360
        self.update_view_display(view_name)
        
    def toggle_overlay(self):
        """Toggle global overlay visibility"""
        self.show_overlay = self.overlay_var.get()
        # Update all individual view overlay checkboxes to match global state
        for view_name in self.views:
            if self.show_overlay:
                # If global is turned on, restore individual view states
                self.overlay_vars[view_name].set(self.views[view_name]['overlay'])
            else:
                # If global is turned off, disable all individual overlays but remember their state
                self.overlay_vars[view_name].set(False)
        self.update_display()
        
    def change_alpha(self, value):
        """Change global overlay alpha"""
        self.overlay_alpha = float(value)
        # Update all individual view alphas to match global alpha
        for view_name in self.views:
            self.views[view_name]['alpha'] = self.overlay_alpha
            self.alpha_vars[view_name].set(self.overlay_alpha)
        self.update_display()
        
    def change_view_alpha(self, value, view_name):
        """Change alpha for a specific view"""
        alpha_value = float(value)
        self.views[view_name]['alpha'] = alpha_value
        self.update_view_display(view_name)
        
    def get_slice_data(self, data, slice_idx, axis=None):
        """Extract slice data based on specified axis"""
        if axis is None:
            axis = 2  # Default to axial
            
        if axis == 0:  # Sagittal
            return data[slice_idx, :, :]
        elif axis == 1:  # Coronal
            return data[:, slice_idx, :]
        else:  # Axial
            return data[:, :, slice_idx]
            
    def normalize_image(self, img_slice):
        """Normalize image slice to 0-255 range"""
        img_min, img_max = np.min(img_slice), np.max(img_slice)
        if img_max > img_min:
            normalized = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img_slice, dtype=np.uint8)
        return normalized
        
    def create_label_overlay(self, label_slice):
        """Create colored overlay from label slice"""
        # Create colormap for labels
        unique_labels = np.unique(label_slice)
        if len(unique_labels) <= 1:  # Only background
            return None
            
        # Use a colormap for different labels
        colormap = plt.get_cmap('tab20')
        overlay = np.zeros((*label_slice.shape, 4), dtype=np.uint8)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            mask = label_slice == label
            color = colormap(label / 20.0)  # Normalize for colormap
            overlay[mask] = [int(c * 255) for c in color]
            
        return overlay
        
    def reset_data(self):
        """Reset all image and label data"""
        self.image_data = None
        self.label_data = None
        
        # Reset slice positions and rotations for all views
        for view_name in self.views:
            self.views[view_name]['slice'] = 0
            self.views[view_name]['rotation'] = 0
        
    def reset_viewer(self):
        """Reset the entire viewer to initial state"""
        self.reset_data()
        
        # Reset UI controls for all views
        for view_name in self.views:
            self.slice_scales[view_name].config(to=100)
            self.slice_vars[view_name].set(0)
            self.slice_labels[view_name].config(text="0/0")
            self.view_visibility_vars[view_name].set(True)
            self.overlay_vars[view_name].set(True)
            self.alpha_vars[view_name].set(0.5)
            self.views[view_name]['show'] = True
            self.views[view_name]['overlay'] = True
            self.views[view_name]['rotation'] = 0
            self.views[view_name]['alpha'] = 0.5
        
        self.overlay_var.set(True)
        self.show_overlay = True
        self.alpha_var.set(0.5)
        self.overlay_alpha = 0.5
        
        # Reset layout
        self.update_view_layout()
        
        # Clear all canvases
        for canvas in self.canvases.values():
            canvas.delete("all")
        
        # Reset status
        self.status_var.set("Ready - Load an image and labels to begin")
        
        # Reset path entries
        self.image_path_var.set("")
        self.label_path_var.set("")
        self.image_path = ""
        self.label_path = ""
        
    def update_image_from_path(self):
        """Load image from the path entered in the input field"""
        filename = self.image_path_var.get().strip()
        if not filename:
            messagebox.showwarning("Warning", "Please enter an image file path")
            return
            
        try:
            # Reset all data when loading a new image
            self.reset_data()
            
            img = nib.load(filename)
            self.image_data = img.get_fdata()
            self.image_path = filename
            self.update_slice_range()
            self.status_var.set(f"Image loaded: {Path(filename).name} - Shape: {self.image_data.shape}")
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def update_labels_from_path(self):
        """Load labels from the path entered in the input field"""
        filename = self.label_path_var.get().strip()
        if not filename:
            messagebox.showwarning("Warning", "Please enter a label file path")
            return
            
        try:
            label_img = nib.load(filename)
            new_label_data = label_img.get_fdata()
            
            # Check if image data exists and if shapes match
            if self.image_data is not None:
                if new_label_data.shape != self.image_data.shape:
                    messagebox.showwarning("Warning", 
                                         f"Label shape {new_label_data.shape} doesn't match image shape {self.image_data.shape}")
                    return  # Don't load mismatched labels
            
            self.label_data = new_label_data
            self.label_path = filename
            unique_labels = np.unique(self.label_data)
            self.status_var.set(f"Labels loaded: {Path(filename).name} - {len(unique_labels)} unique labels")
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load labels: {str(e)}")
        
    def update_display(self):
        """Update all view displays"""
        for view_name in self.views:
            self.update_view_display(view_name)
            
    def update_view_display(self, view_name):
        """Update display for a specific view"""
        if self.image_data is None:
            return
            
        canvas = self.canvases[view_name]
        view_data = self.views[view_name]
        
        # Clear canvas first
        canvas.delete("all")
        
        # If view is not shown, keep canvas black
        if not view_data['show']:
            return
            
        # Get current slice for this view
        axis = view_data['axis']
        slice_idx = view_data['slice']
        img_slice = self.get_slice_data(self.image_data, slice_idx, axis)
        
        # Normalize image
        img_normalized = self.normalize_image(img_slice)
        
        # Create base image (grayscale -> RGB)
        img_rgb = np.stack([img_normalized] * 3, axis=-1)
        
        # Add label overlay if available and enabled (both global and view-specific)
        if (self.label_data is not None and self.show_overlay and view_data['overlay']):
            label_slice = self.get_slice_data(self.label_data, slice_idx, axis)
            overlay = self.create_label_overlay(label_slice)
            
            if overlay is not None:
                # Blend overlay with image using view-specific alpha
                alpha = view_data['alpha']
                mask = overlay[:, :, 3] > 0  # Non-transparent pixels
                
                for c in range(3):  # RGB channels
                    img_rgb[mask, c] = (1 - alpha) * img_rgb[mask, c] + alpha * overlay[mask, c]
        
        # Convert to PIL Image and display
        # Flip vertically for proper medical image orientation
        img_rgb = np.flipud(img_rgb)
        pil_image = Image.fromarray(img_rgb.astype(np.uint8))
        
        # Apply rotation if specified
        rotation = view_data['rotation']
        if rotation != 0:
            pil_image = pil_image.rotate(-rotation, expand=True)
        
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
            img_aspect = pil_image.width / pil_image.height
            canvas_aspect = canvas_width / canvas_height
            
            if img_aspect > canvas_aspect:
                new_width = canvas_width - 20
                new_height = int(new_width / img_aspect)
            else:
                new_height = canvas_height - 20
                new_width = int(new_height * img_aspect)
                
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Update canvas
        photo = ImageTk.PhotoImage(pil_image)
        # Store reference to prevent garbage collection
        setattr(self, f'{view_name}_photo', photo)
        canvas.create_image(canvas_width//2, canvas_height//2, 
                           image=photo, anchor=tk.CENTER)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = NIfTIViewer(root)
    
    # Bind resize event to update display for all canvases
    def on_canvas_configure(event):
        app.update_display()
    
    for canvas in app.canvases.values():
        canvas.bind('<Configure>', on_canvas_configure)
    
    root.mainloop()


if __name__ == "__main__":
    main()