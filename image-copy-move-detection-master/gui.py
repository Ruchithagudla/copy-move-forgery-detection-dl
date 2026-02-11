import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Suppress OpenCV warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import threading
import shutil
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from copy_move_detection.ml_classifier import ImageClassifier

class DatasetOrganizer:
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        
    def organize(self, src_folder, dest_folder, mode='auto'):
        orig_path = os.path.join(dest_folder, 'original')
        forged_path = os.path.join(dest_folder, 'forged')
        
        os.makedirs(orig_path, exist_ok=True)
        os.makedirs(forged_path, exist_ok=True)
        
        valid_files = [f for f in os.listdir(src_folder) 
                     if f.lower().endswith(('.jpg','.jpeg','.png'))]
        total_files = len(valid_files)
        
        for i, filename in enumerate(valid_files):
            src_path = os.path.join(src_folder, filename)
            
            try:
                if mode == 'auto':
                    dest = self._auto_classify(filename)
                elif mode == 'manual':
                    dest = self._manual_classify(src_path, filename)
                else:
                    dest = self._checksum_classify(src_path, orig_path)
                
                shutil.copy2(src_path, os.path.join(dest, filename))
                self._report_progress(i+1, total_files, filename, dest)
                
            except Exception as e:
                self._report_error(filename, str(e))
        
        self._report_completion(orig_path, forged_path)

    def _auto_classify(self, filename):
        forged_keywords = ['forged', 'tampered', 'copy', 'edited', 'fake']
        return 'forged' if any(kw in filename.lower() for kw in forged_keywords) else 'original'
    
    def _manual_classify(self, img_path, filename):
        popup = tk.Toplevel()
        popup.title("Classify Image")
        popup.geometry("300x200")
        
        img = Image.open(img_path)
        img.load()  # Handle PNG warnings
        img.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(img)
        
        tk.Label(popup, image=photo).pack()
        tk.Label(popup, text=filename).pack()
        
        choice = tk.StringVar(value='original')
        tk.Radiobutton(popup, text="Original", variable=choice, value='original').pack()
        tk.Radiobutton(popup, text="Forged", variable=choice, value='forged').pack()
        
        def confirm():
            popup.destroy()
        
        tk.Button(popup, text="Confirm", command=confirm).pack()
        
        popup.wait_window()
        return choice.get()

    def _report_progress(self, current, total, filename, destination):
        if self.status_callback:
            self.status_callback(
                f"Processing {current}/{total}: {filename[:15]}... → {destination}",
                int(100 * current/total)
            )

    def _report_error(self, filename, error):
        if self.status_callback:
            self.status_callback(f"Error with {filename}: {error}", -1)

    def _report_completion(self, orig_path, forged_path):
        if self.status_callback:
            orig_count = len(os.listdir(orig_path))
            forged_count = len(os.listdir(forged_path))
            self.status_callback(
                f"Complete! {orig_count} originals, {forged_count} forged",
                100
            )

class CopyMoveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Copy-Move Forgery Detector")
        self.root.geometry("1000x800")
        
        self.classifier = ImageClassifier()
        self.organizer = DatasetOrganizer()
        self.training_in_progress = False
        self.organizing_in_progress = False
        
        self.setup_ui()
    
    def setup_ui(self):
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Image Display Frames
        self.original_frame = tk.LabelFrame(self.root, text="Original Image")
        self.original_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas_original = tk.Label(self.original_frame)
        self.canvas_original.pack()
        
        self.result_frame = tk.LabelFrame(self.root, text="Detection Result")
        self.result_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.canvas_result = tk.Label(self.result_frame)
        self.canvas_result.pack()
        
        # Control Buttons
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.btn_open = tk.Button(self.btn_frame, text="Open Image", 
                                command=self.open_image, width=15)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        self.btn_train = tk.Button(self.btn_frame, text="Train Model", 
                                 command=self.initiate_training, width=15)
        self.btn_train.pack(side=tk.LEFT, padx=5)
        
        self.btn_classify = tk.Button(self.btn_frame, text="Classify", 
                                    command=self.classify_image, width=15,
                                    state=tk.DISABLED)
        self.btn_classify.pack(side=tk.LEFT, padx=5)
        
        self.btn_organize = tk.Button(self.btn_frame, text="Organize Dataset", 
                                    command=self.initiate_organization, width=15)
        self.btn_organize.pack(side=tk.LEFT, padx=5)
        
        # Progress Bars
        self.training_progress = ttk.Progressbar(self.root, orient="horizontal")
        self.training_progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
        
        self.organization_progress = ttk.Progressbar(self.root, orient="horizontal")
        self.organization_progress.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10)
        
        # Status Labels
        self.training_status = tk.Label(self.root, text="Ready for training", anchor=tk.W)
        self.training_status.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10)
        
        self.organization_status = tk.Label(self.root, text="Ready for organization", anchor=tk.W)
        self.organization_status.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10)
        
        # Metrics Frame
        self.metrics_frame = tk.LabelFrame(self.root, text="Model Metrics")
        self.metrics_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.accuracy_label = tk.Label(self.metrics_frame, text="Accuracy: N/A")
        self.accuracy_label.pack()
        
        self.confusion_canvas = None
    
    def verify_dataset_structure(self, dataset_path):
        """Verify dataset has both original and forged folders with images"""
        required = ['original', 'forged']
        for folder in required:
            folder_path = os.path.join(dataset_path, folder)
            if not os.path.exists(folder_path):
                return False
            if len(os.listdir(folder_path)) == 0:
                return False
        return True
    
    def open_image(self):
        file_types = [("Image files", "*.png;*.jpg;*.jpeg")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            try:
                img = Image.open(file_path)
                img.load()  # Handle PNG warnings
                self.current_image = file_path
                self.display_image(file_path, self.canvas_original)
                self.training_status.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.btn_classify.config(state=tk.NORMAL if self.classifier.trained else tk.DISABLED)
            except Exception as e:
                self.training_status.config(text=f"Error loading image: {str(e)}")
    
    def display_image(self, path, canvas):
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))
            self.tk_img = ImageTk.PhotoImage(img)
            canvas.config(image=self.tk_img)
        except Exception as e:
            raise ValueError(f"Could not display image: {str(e)}")
    
    def initiate_training(self):
        if self.training_in_progress:
            self.training_status.config(text="Training already in progress")
            return
            
        dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if not dataset_path:
            return
            
        if not self.verify_dataset_structure(dataset_path):
            self.training_status.config(text="Error: Need 'original' and 'forged' folders with images")
            return
            
        self.training_in_progress = True
        self.training_status.config(text="Preparing training...")
        self.training_progress["value"] = 0
        self.btn_train.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        
        threading.Thread(
            target=self.train_model,
            args=(dataset_path,),
            daemon=True
        ).start()
    
    def train_model(self, dataset_path):
        try:
            # Reset features and labels for new training
            self.classifier.features = []
            self.classifier.labels = []
            
            total_files = 0
            for root, _, files in os.walk(dataset_path):
                total_files += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if total_files == 0:
                raise ValueError("No valid images found in dataset")
            
            processed = 0
            self.root.after(0, lambda: self.training_progress.config(maximum=total_files))
            
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        label = 1 if 'forged' in root.lower() else 0
                        
                        if self.classifier.extract_features(file_path, label):
                            processed += 1
                            self.root.after(0, lambda p=processed, t=total_files: [
                                self.training_progress.config(value=p),
                                self.training_status.config(text=f"Processing {p}/{t} images")
                            ])
            
            # Verify we have both classes before training
            if len(np.unique(self.classifier.labels)) < 2:
                raise ValueError("Dataset must contain both original and forged images")
            
            accuracy, cm = self.classifier.train_model()
            self.root.after(0, lambda: [
                self.accuracy_label.config(text=f"Accuracy: {accuracy:.2%}"),
                self.btn_classify.config(state=tk.NORMAL),
                self.training_status.config(text=f"Training complete! Accuracy: {accuracy:.2%}"),
                self.show_confusion_matrix(cm)
            ])
            
        except Exception as e:
            error_msg = str(e)  # Capture error message for lambda
            self.root.after(0, lambda msg=error_msg: [
                self.training_status.config(text=f"Training failed: {msg}", fg="red"),
                self.btn_train.config(state=tk.NORMAL),
                self.btn_open.config(state=tk.NORMAL)
            ])
        finally:
            self.training_in_progress = False
    
    def classify_image(self):
        if not hasattr(self, 'current_image'):
            self.training_status.config(text="Please open an image first!")
            return
            
        try:
            self.training_status.config(text="Classifying...", fg="black")
            prediction, confidence = self.classifier.predict_image(self.current_image)
            
            img = Image.open(self.current_image)
            img.thumbnail((400, 400))
            
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), 
                     f"{prediction} ({confidence:.1%})",
                     fill="red" if prediction == "Forged" else "green",
                     font=font,
                     stroke_width=2,
                     stroke_fill="black")
            
            self.result_img = ImageTk.PhotoImage(img)
            self.canvas_result.config(image=self.result_img)
            self.training_status.config(text=f"Result: {prediction} (Confidence: {confidence:.1%})")
            
        except Exception as e:
            self.training_status.config(text=f"Classification error: {str(e)}", fg="red")
    
    def show_confusion_matrix(self, cm):
        if self.confusion_canvas:
            self.confusion_canvas.get_tk_widget().destroy()
        
        fig = plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classifier.class_names,
                   yticklabels=self.classifier.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        self.confusion_canvas = FigureCanvasTkAgg(fig, master=self.metrics_frame)
        self.confusion_canvas.draw()
        self.confusion_canvas.get_tk_widget().pack()
    
    def initiate_organization(self):
        if self.organizing_in_progress:
            messagebox.showwarning("Warning", "Organization already in progress")
            return
            
        src_folder = filedialog.askdirectory(title="Select Source Folder")
        if not src_folder:
            return
            
        dest_folder = filedialog.askdirectory(title="Select Target Folder")
        if not dest_folder:
            return
            
        mode = self._select_organization_mode()
        if not mode:
            return
            
        self.organizing_in_progress = True
        self.organization_status.config(text="Preparing...")
        self.organization_progress["value"] = 0
        self.btn_organize.config(state=tk.DISABLED)
        
        threading.Thread(
            target=self._run_organization,
            args=(src_folder, dest_folder, mode),
            daemon=True
        ).start()
        
    def _select_organization_mode(self):
        popup = tk.Toplevel(self.root)
        popup.title("Organization Mode")
        
        mode = tk.StringVar(value='auto')
        
        ttk.Radiobutton(popup, text="Automatic (filename/EXIF)", 
                       variable=mode, value='auto').pack(anchor=tk.W, padx=20, pady=5)
        ttk.Radiobutton(popup, text="Manual Classification", 
                       variable=mode, value='manual').pack(anchor=tk.W, padx=20, pady=5)
        
        def confirm():
            popup.destroy()
            
        ttk.Button(popup, text="Start", command=confirm).pack(pady=10)
        
        popup.wait_window()
        return mode.get() if popup.winfo_exists() else None
        
    def _run_organization(self, src, dest, mode):
        try:
            organizer = DatasetOrganizer(status_callback=self._update_organization_status)
            organizer.organize(src, dest, mode)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: [
                self.btn_organize.config(state=tk.NORMAL),
                setattr(self, 'organizing_in_progress', False)
            ])
            
    def _update_organization_status(self, message, percent):
        self.root.after(0, lambda: [
            self.organization_status.config(text=message),
            self.organization_progress.config(value=percent if percent >=0 else 0)
        ])

if __name__ == "__main__":
    root = tk.Tk()
    app = CopyMoveApp(root)
    root.mainloop()