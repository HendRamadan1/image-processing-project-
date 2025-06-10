import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog, colorchooser
import Filtring
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Global variables
colors = {
    'primary': '#6d8ed4',
    'secondary': '#4a7f9d',
    'accent': '#2d4d50',
    'background': '#ffffff',
    'text': '#2d4d50'
}

font_regular = ('Raleway', 10)
font_bold = ('Raleway', 10, 'bold')
font_title = ('Raleway', 14, 'bold')

pen_color = colors['accent']
pen_size = 3
file_path = ""


root = tk.Tk()
root.title("Project | Image Processing")
root.geometry('1700x900')  
root.config(bg=colors['background'])


container = tk.Frame(root, bg=colors['background'])
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container, bg=colors['background'], highlightthickness=0)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

scrollable_frame = tk.Frame(canvas, bg=colors['background'])
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

# UI Content
title_frame = tk.Frame(scrollable_frame, bg=colors['primary'])
title_frame.pack(fill='x')

tk.Label(title_frame, 
        text="Welcome in our Image Processing project", 
        bg=colors['primary'],
        fg='white',
        font=font_bold).pack(side='left', padx=10)

tk.Label(title_frame, 
        text="IMAGE PROCESSING", 
        bg=colors['primary'],
        fg='white',
        font=font_title).pack(side='right', padx=10)

main_frame = tk.Frame(scrollable_frame, bg=colors['background'])
main_frame.pack(fill='both', expand=True, padx=20, pady=20)


sidebar = tk.Frame(main_frame, bg=colors['background'], width=250)
sidebar.pack(side='left', fill='y', padx=(0, 20))

tk.Label(sidebar, 
        text="Image Processing Tool", 
        bg=colors['background'],
        fg=colors['text'],
        font=font_title).pack(pady=(0, 20))


img_frame = tk.Frame(main_frame, bg=colors['background'])
img_frame.pack(side='right', fill='both', expand=True)


original_frame = tk.Frame(
    img_frame, 
    bg=colors['background']
)
original_frame.pack(
    side='left', 
    fill='both', 
    expand=True, 
    padx=10
)


tk.Label(
    original_frame,
    text="original image",
    bg=colors['background'],
    fg=colors['text'],
    font=font_bold
).pack()
canvas1 = tk.Canvas(
    original_frame,
    width=550,
    height=450,
    bg='white',
    highlightthickness=1,
    highlightbackground=colors['secondary']
)
canvas1.pack()


original_hist_frame = tk.Frame(
    original_frame,
    bg=colors['background']
)
original_hist_frame.pack(
    side='bottom',
    fill='both',
    expand=True,
    padx=10
)

tk.Label(
    original_hist_frame,
    text="original histogram",
    bg=colors['background'],
    fg=colors['text'],
    font=font_bold
).pack()

original_hist_canvas = tk.Canvas(
    original_hist_frame,
    width=550,
    height=450,
    bg='white',
    highlightthickness=1,
    highlightbackground=colors['secondary']
)
original_hist_canvas.pack()



processed_frame = tk.Frame(
    img_frame,
    bg=colors['background']
)
processed_frame.pack(
    side='right',
    fill='both',
    expand=True,
    padx=10
)


tk.Label(
    processed_frame,
    text="Processed Image",
    bg=colors['background'],
    fg=colors['text'],
    font=font_bold
).pack()

canvas2 = tk.Canvas(
    processed_frame,
    width=550,
    height=450,
    bg='white',
    highlightthickness=1,
    highlightbackground=colors['secondary']
)
canvas2.pack()


processed_hist_frame = tk.Frame(
    processed_frame,
    bg=colors['background']
)
processed_hist_frame.pack(
    side='bottom',
    fill='both',
    expand=True,
    padx=10
)

tk.Label(
    processed_hist_frame,
    text="Processed Histogram",
    bg=colors['background'],
    fg=colors['text'],
    font=font_bold
).pack()

processed_hist_canvas = tk.Canvas(
    processed_hist_frame,
    width=550,
    height=450,
    bg='white',
    highlightthickness=1,
    highlightbackground=colors['secondary']
)
processed_hist_canvas.pack()


def create_button(parent, text, command):
    btn = tk.Button(parent,
                text=text,
                command=command,
                bg=colors['primary'],
                fg='white',
                font=font_bold,
                borderwidth=0,
                padx=10,
                pady=8,
                activebackground=colors['secondary'],
                activeforeground='white')
    btn.pack(fill='x', pady=5)
    return btn

def display_histogram(image, canvas,  threshold=None):
    """Display histogram with optional threshold line"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if hasattr(canvas, 'figure'):
        canvas.figure.clf()
        plt.close(canvas.figure)
    
    figure = plt.figure(figsize=(5.5, 4.5), dpi=100)
    ax = figure.add_subplot(111)
    
    if len(img_array.shape) == 2:  
        
        ax.hist(img_array.ravel(), bins=250, range=(0, 255), color='gray',)
        ax.set_facecolor('#f5f5f5')
        
        
        if threshold is not None:
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold = {threshold:.2f}')
            ax.legend()
    else:  
        colors = ('red', 'green', 'blue')
        for i, color in enumerate(colors):
            ax.hist(img_array[:, :, i].ravel(), bins=256, range=(0, 256), 
                   color=color, alpha=0.5)
        ax.set_facecolor('#f5f5f5')


    ax.set_xlim([0, 256])
    ax.set_xticks(np.arange(0, 256, 50))
    ax.set_xticklabels(np.arange(0, 256, 50), fontsize=8)
    
    max_count = np.histogram(img_array.ravel(), bins=256, range=(0, 256))[0].max()
    y_ticks = np.linspace(0, max_count, 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y):,}" for y in y_ticks], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    for widget in canvas.winfo_children():
        widget.destroy()
    canvas.figure = figure
    histogram_canvas = FigureCanvasTkAgg(figure, master=canvas)
    histogram_canvas.draw()
    histogram_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
    
    
    
    

def add_image():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((550, 450))
        display_image(image, canvas1)
        display_histogram(image, original_hist_canvas)

def display_image(image, canvas):
    photo = ImageTk.PhotoImage(image)
    canvas.image = photo
    canvas.delete("all")
    canvas.create_image(0, 0, anchor='nw', image=photo)

def draw(event):
    x1, y1 = (event.x - pen_size), (event.y - pen_size)
    x2, y2 = (event.x + pen_size), (event.y + pen_size)
    canvas2.create_oval(x1, y1, x2, y2, fill=pen_color, outline='')

def change_pen_color():
    global pen_color
    color = colorchooser.askcolor(title='Choose Pen Color')
    if color[1]:
        pen_color = color[1]

def clear_photo():
    canvas2.delete('all')
    if hasattr(canvas2, 'image'):
        canvas2.create_image(0, 0, anchor='nw', image=canvas2.image)
    processed_hist_canvas.delete('all')

def ApplyAddImage(value):
    if file_path:
        image = Filtring.AddPhoto(file_path, value)
        display_processed_image(image)
        display_histogram(image, original_hist_canvas)

def ApplySubtractionImage(value):
    if file_path:
        image = Filtring.SubtractionPhoto(file_path, value)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas)
        

def ApplyDivisionImage(value):
    if file_path and value:
        try:
            value = int(value)
            if value == 0:
                value = 1
            image = Filtring.DivisionPhoto(file_path, value)
            display_processed_image(image)
        except ValueError:
            pass

def ApplyRedColor(value):
    if file_path:
        image = Filtring.AddRedColor(file_path, value)
        display_processed_image(image)

def display_processed_image(image, filter_name=None, threshold=None):
    image = image.resize((550, 450))
    photo = ImageTk.PhotoImage(image)
    canvas2.image = photo
    canvas2.delete("all")
    canvas2.create_image(0, 0, anchor='nw', image=photo)
    display_histogram(image, processed_hist_canvas, threshold)

def apply_filter_gray(filter_type):
    if file_path:
        image = Image.open(file_path)
        if filter_type == 'Gray':
            image = Filtring.Gray(file_path)
        elif filter_type == 'SwapGreenRed':
            image = Filtring.SwapGreenRedColor(file_path)
        elif filter_type == 'Complement':
            image = Filtring.ComplementPhoto(file_path)
        elif filter_type == 'Stretching':
            image = Filtring.Stretching(file_path)
        elif filter_type == 'EliminateRed':
            image = Filtring.EliminateRed(file_path)
        elif filter_type == 'Equalization':
            image = Filtring.Equalization(file_path)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas)

def apply_filter(filter_type):
    if file_path:
        image = Image.open(file_path)
        if filter_type == 'AverageFilter':
            image = Filtring.AverageFilter(file_path)
        elif filter_type == 'LaplacianFilter':
            image = Filtring.LaplacianFilter(file_path)
        elif filter_type == 'MaximumFilter':
            image = Filtring.MaximumFilter(file_path)
        elif filter_type == 'MinimumFilter':
            image = Filtring.MinimumFilter(file_path)
        elif filter_type == 'MedianFilter':
            image = Filtring.MedianFilter(file_path)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas)

def apply_salt_pepper(option):
    if file_path:
        if option == 'AddNoise':
            image = Filtring.AddSaltPepperNoise(file_path)
        elif option == 'Restore_Average':
            image = Filtring.RestoreWithAverage(file_path)
        elif option == 'Restore_Median':
            image = Filtring.RestoreWithMedian(file_path)
        elif option == 'Restore_Outlier':
            image = Filtring.RestoreWithOutlier(file_path, threshold=0.4)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas)

def apply_gaussian_filter(filter_type):
    if file_path:
        if filter_type == 'AddGaussianNoise':
            image = Filtring.AddGaussianNoise(file_path)
        elif filter_type == 'Restore_ImageAveraging':
            image = Filtring.RestoreByImageAveraging(file_path)
        elif filter_type == 'Restore_AverageFilter':
            image = Filtring.RestoreByAverageFilter(file_path)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas)

def apply_segmentation(method):
    if file_path:
        if method == 'BasicGlobal':
            image = Filtring.BasicGlobalThresholding(file_path)
        elif method == 'Automatic':
            image, threshold = Filtring.AutomaticThresholding(file_path, return_threshold=True) 
        elif method == 'Adaptive':
            image = Filtring.AdaptiveThresholding(file_path)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas,threshold)

def apply_edge_detection(method):
    if file_path and method == 'Sobel':
        image = Filtring.SobelEdgeDetection(file_path)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas)

def apply_morphology(operation):
    if file_path:
        if operation == 'Dilation':
            image = Filtring.Dilation(file_path)
        elif operation == 'Erosion':
            image = Filtring.Erosion(file_path)
        elif operation == 'Opening':
            image = Filtring.Opening(file_path)
        elif operation == 'InternalBoundary':
            image = Filtring.InternalBoundary(file_path)
        elif operation == 'ExternalBoundary':
            image = Filtring.ExternalBoundary(file_path)
        elif operation == 'MorphGradient':
            image = Filtring.MorphGradient(file_path)
        display_processed_image(image)
        display_histogram(image, processed_hist_canvas)
        


create_button(sidebar, " Load Image", add_image)


drawing_frame = tk.LabelFrame(sidebar,
                            text="Drawing Tools",
                            bg=colors['background'],
                            fg=colors['text'],
                            font=font_bold,
                            padx=10,
                            pady=10)
drawing_frame.pack(fill='x', pady=10)

create_button(drawing_frame, " Pen Color", change_pen_color)

size_frame = tk.Frame(drawing_frame, bg=colors['background'])
size_frame.pack(fill='x', pady=5)

tk.Label(size_frame, 
        text="Pen Size:", 
        bg=colors['background'],
        fg=colors['text'],
        font=font_regular).pack(side='left')

pen_size_var = tk.IntVar(value=3)
sizes = [("Small", 2), ("Medium", 5), ("Large", 8)]

for text, size in sizes:
    rb = tk.Radiobutton(size_frame,
                    text=text,
                    value=size,
                    variable=pen_size_var,
                    bg=colors['background'],
                    fg=colors['text'],
                    font=font_regular,
                    selectcolor=colors['primary'])
    rb.pack(side='left', padx=5)

create_button(drawing_frame, "🧹 Clear Canvas", clear_photo)


operations_frame = tk.LabelFrame(sidebar,
                            text="Basic Operations",
                            bg=colors['background'],
                            fg=colors['text'],
                            font=font_bold,
                            padx=10,
                            pady=10)
operations_frame.pack(fill='x', pady=10)


def create_operation_field(parent, label, command):
    frame = tk.Frame(parent, bg=colors['background'])
    frame.pack(fill='x', pady=3)
    
    tk.Label(frame, 
            text=label, 
            bg=colors['background'],
            fg=colors['text'],
            font=font_regular).pack(side='left')
    
    entry = tk.Entry(frame, 
                    width=8,
                    bg='white',
                    fg=colors['text'],
                    font=font_regular,
                    borderwidth=1,
                    relief='solid')
    entry.pack(side='left', padx=5)
    
    btn = tk.Button(frame,
                text="Apply",
                command=lambda: globals()[command](entry.get()),
                bg=colors['secondary'],
                fg='white',
                font=font_bold,
                borderwidth=0,
                padx=8,
                pady=2)
    btn.pack(side='left')

create_operation_field(operations_frame, " Add Value:", "ApplyAddImage")
create_operation_field(operations_frame, " Subtract Value:", "ApplySubtractionImage")
create_operation_field(operations_frame, " Divide By:", "ApplyDivisionImage")
create_operation_field(operations_frame, " Red Channel:", "ApplyRedColor")

# Filters section
filters_frame = tk.LabelFrame(sidebar,
                            text="Filters & Effects",
                            bg=colors['background'],
                            fg=colors['text'],
                            font=font_bold,
                            padx=10,
                            pady=10)
filters_frame.pack(fill='x', pady=10)

def create_combobox(parent, label, values, command):
    frame = tk.Frame(parent, bg=colors['background'])
    frame.pack(fill='x', pady=3)
    
    tk.Label(frame, 
            text=label, 
            bg=colors['background'],
            fg=colors['text'],
            font=font_regular).pack(side='top', anchor='w')
    
    cb = ttk.Combobox(frame, 
                    values=values, 
                    state='readonly',
                    font=font_regular)
    cb.pack(fill='x', pady=2)
    cb.bind('<<ComboboxSelected>>', lambda e: globals()[command](cb.get()))

create_combobox(filters_frame, "Color Filters:", 
            ['Complement', 'SwapGreenRed', 'EliminateRed', 
                'Gray', 'Stretching', 'Equalization'],
            'apply_filter_gray')

create_combobox(filters_frame, "Image Filters:", 
            ['AverageFilter', 'LaplacianFilter', 'MaximumFilter', 
                'MinimumFilter', 'MedianFilter'],
            'apply_filter')

# Noise section
noise_frame = tk.LabelFrame(sidebar,
                        text="Noise & Restoration",
                        bg=colors['background'],
                        fg=colors['text'],
                        font=font_bold,
                        padx=10,
                        pady=10)
noise_frame.pack(fill='x', pady=10)

create_combobox(noise_frame, "Salt & Pepper:", 
            ['AddNoise', 'Restore_Average', 
                'Restore_Median', 'Restore_Outlier'],
            'apply_salt_pepper')

create_combobox(noise_frame, "Gaussian Noise:", 
            ['AddGaussianNoise', 'Restore_ImageAveraging', 
                'Restore_AverageFilter'],
            'apply_gaussian_filter')

# Advanced section
advanced_frame = tk.LabelFrame(sidebar,
                            text="Advanced Operations",
                            bg=colors['background'],
                            fg=colors['text'],
                            font=font_bold,
                            padx=10,
                            pady=10)
advanced_frame.pack(fill='x', pady=10)

create_combobox(advanced_frame, "Segmentation:", 
            ['BasicGlobal', 'Automatic', 'Adaptive'],
            'apply_segmentation')

create_combobox(advanced_frame, "Edge Detection:", 
            ['Sobel'],
            'apply_edge_detection')


# Bind drawing event
canvas2.bind("<B1-Motion>", draw)

root.mainloop()