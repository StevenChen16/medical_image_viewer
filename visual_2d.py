import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors
from tkinter import Tk, filedialog
import os

# Set default font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def select_image_file():
    Tk().withdraw()
    return filedialog.askopenfilename(title='Select Image File (.mha)', filetypes=[('MHA Files', '*.mha')])

def select_label_file():
    Tk().withdraw()
    return filedialog.askopenfilename(title='Select Label File (.mha)', filetypes=[('MHA Files', '*.mha')])

def get_default_label_path(image_path):
    if "task1" in image_path:
        return image_path.replace("ImagesTr", "LabelsTr").rsplit('_', 1)[0] + ".mha"
    elif "task2" in image_path:
        return image_path.replace("ImagesTr", "LabelsTr")
    return ""

def visualize_2d_with_labels():
    image_path = select_image_file()
    if not image_path:
        print("Image file selection cancelled.")
        return

    default_label_path = get_default_label_path(image_path)
    if os.path.exists(default_label_path):
        label_path = default_label_path
    else:
        label_path = select_label_file()
        if not label_path:
            print("Label file selection cancelled.")
            return

    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label)

    print(f"Image shape: {image_array.shape}")
    print(f"Label shape: {label_array.shape}")
    print(f"Image range: [{np.min(image_array)}, {np.max(image_array)}]")
    print(f"Label values: {np.unique(label_array)}")

    # Create figure with larger image area
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.45, left=0.1, right=0.9, top=0.9)

    z_slice = [image_array.shape[0] // 2]
    img_display = ax.imshow(image_array[z_slice[0], :, :], cmap='gray')

    colors = ['none', 'red', 'blue']
    cmap = mcolors.ListedColormap(colors[1:])
    bounds = [0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    mask = np.ma.masked_where(label_array[z_slice[0], :, :] == 0, label_array[z_slice[0], :, :])
    label_display = ax.imshow(mask, cmap=cmap, norm=norm, alpha=0.5)

    ax.set_title(f'Slice #{z_slice[0]} - Red: Tumor, Blue: Pancreas')

    # Move UI controls to create more space for the image
    ax_slider = plt.axes([0.25, 0.3, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, image_array.shape[0]-1, valinit=z_slice[0], valstep=1)

    ax_ww = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_wl = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_visibility = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_opacity = plt.axes([0.25, 0.1, 0.65, 0.03])

    # Position buttons vertically instead of horizontally
    # First button
    ax_image_select = plt.axes([0.05, 0.9, 0.15, 0.05])
    # Second button below the first one
    ax_label_select = plt.axes([0.05, 0.84, 0.15, 0.05])
    
    ax_legend = plt.axes([0.025, 0.025, 0.95, 0.05])
    ax_legend.axis('off')
    ax_legend.text(0, 0.8, "[Shortcuts] Up/Down: Slice | Left/Right: Window Level | M: Toggle Mask", fontsize=10)

    ww_init = np.max(image_array) // 2
    wl_init = np.max(image_array) // 4

    slider_ww = Slider(ax_ww, 'Window Width', 1, np.max(image_array), valinit=ww_init)
    slider_wl = Slider(ax_wl, 'Window Level', 0, np.max(image_array), valinit=wl_init)
    slider_visibility = Slider(ax_visibility, 'Label Visibility', 0, 1, valinit=1, valstep=1)
    slider_opacity = Slider(ax_opacity, 'Label Opacity', 0, 1, valinit=0.5)
    button_img = Button(ax_image_select, 'Select Image')
    button_lbl = Button(ax_label_select, 'Select Label')

    def update_display():
        z = z_slice[0]
        img_display.set_data(image_array[z, :, :])
        mask = np.ma.masked_where(label_array[z, :, :] == 0, label_array[z, :, :])
        label_display.set_data(mask)
        label_display.set_visible(slider_visibility.val == 1)
        label_display.set_alpha(slider_opacity.val)
        ax.set_title(f'Slice #{z} - Red: Tumor, Blue: Pancreas')
        fig.canvas.draw_idle()

    def update(val):
        z_slice[0] = int(slider.val)
        update_display()

    def update_window(val):
        ww = slider_ww.val
        wl = slider_wl.val
        vmin = wl - ww/2
        vmax = wl + ww/2
        img_display.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    def update_label_visibility(val):
        label_display.set_visible(slider_visibility.val == 1)
        fig.canvas.draw_idle()

    def update_label_opacity(val):
        label_display.set_alpha(slider_opacity.val)
        fig.canvas.draw_idle()

    def reload_image(event):
        new_image_path = select_image_file()
        if new_image_path:
            default_label_path = get_default_label_path(new_image_path)
            if os.path.exists(default_label_path):
                new_label_path = default_label_path
            else:
                new_label_path = select_label_file()
                if not new_label_path:
                    print("Label file selection cancelled.")
                    return
            
            nonlocal image, label, image_array, label_array
            image = sitk.ReadImage(new_image_path)
            label = sitk.ReadImage(new_label_path)
            image_array = sitk.GetArrayFromImage(image)
            label_array = sitk.GetArrayFromImage(label)
            
            print(f"Image shape: {image_array.shape}")
            print(f"Label shape: {label_array.shape}")
            print(f"Image range: [{np.min(image_array)}, {np.max(image_array)}]")
            print(f"Label values: {np.unique(label_array)}")
            
            z_slice[0] = image_array.shape[0] // 2
            
            ax_slider.clear()
            slider.__dict__.clear()
            slider.__init__(ax_slider, 'Slice', 0, image_array.shape[0]-1, valinit=z_slice[0], valstep=1)
            slider.on_changed(update)
            
            ww_init = np.max(image_array) // 2
            wl_init = np.max(image_array) // 4
            slider_ww.set_val(ww_init)
            slider_wl.set_val(wl_init)
            
            ax.set_title(f'Slice #{z_slice[0]} - Red: Tumor, Blue: Pancreas')
        
            update_display()
            update_window(None)

    def reload_label(event):
        new_label_path = select_label_file()
        if new_label_path:
            nonlocal label, label_array
            label = sitk.ReadImage(new_label_path)
            label_array = sitk.GetArrayFromImage(label)
            
            print(f"Label shape: {label_array.shape}")
            print(f"Label values: {np.unique(label_array)}")
            
            if label_array.shape[0] != image_array.shape[0]:
                print(f"Warning: Label dimensions ({label_array.shape}) don't match image dimensions ({image_array.shape})!")
                new_max = min(image_array.shape[0], label_array.shape[0]) - 1
                if z_slice[0] > new_max:
                    z_slice[0] = new_max // 2
                
                ax_slider.clear()
                slider.__dict__.clear()
                slider.__init__(ax_slider, 'Slice', 0, new_max, valinit=z_slice[0], valstep=1)
                slider.on_changed(update)
            
            update_display()

    def on_key(event):
        if event.key == 'up':
            z_slice[0] = min(z_slice[0] + 1, image_array.shape[0] - 1)
            slider.set_val(z_slice[0])
        elif event.key == 'down':
            z_slice[0] = max(z_slice[0] - 1, 0)
            slider.set_val(z_slice[0])
        elif event.key == 'left':
            slider_wl.set_val(max(slider_wl.val - 10, slider_wl.valmin))
        elif event.key == 'right':
            slider_wl.set_val(min(slider_wl.val + 10, slider_wl.valmax))
        elif event.key == 'm':
            current = slider_visibility.val
            slider_visibility.set_val(1 - current)

    slider.on_changed(update)
    slider_ww.on_changed(update_window)
    slider_wl.on_changed(update_window)
    slider_visibility.on_changed(update_label_visibility)
    slider_opacity.on_changed(update_label_opacity)
    button_img.on_clicked(reload_image)
    button_lbl.on_clicked(reload_label)

    fig.canvas.mpl_connect('key_press_event', on_key)

    vmin = wl_init - ww_init/2
    vmax = wl_init + ww_init/2
    img_display.set_clim(vmin, vmax)

    plt.show()

visualize_2d_with_labels()