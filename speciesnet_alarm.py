import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import cv2
import os
import numpy as np
from PIL import Image, ImageTk, ImageColor
import tempfile
import torch
import time
import threading
import sys
import winsound  # For Windows alarm sound
import platform  # To detect OS

# Attempt to download/import SpeciesNet just like the original script
try:
    import kagglehub

    # Download and import SpeciesNet from Kaggle Hub
    print("Downloading SpeciesNet model from Kaggle Hub...")
    model_path = kagglehub.model_download("google/speciesnet/pyTorch/v4.0.1b")
    print(f"Model downloaded to: {model_path}")

    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    from speciesnet import (
        DEFAULT_MODEL,
        draw_bboxes,
        SpeciesNet,
        SUPPORTED_MODELS,
    )

    KAGGLE_DOWNLOAD_SUCCESS = True
    KAGGLE_MODEL_PATH = model_path

except Exception as e:
    print(f"Error downloading from Kaggle Hub: {e}")
    print("Falling back to local SpeciesNet installation...")
    KAGGLE_DOWNLOAD_SUCCESS = False
    KAGGLE_MODEL_PATH = None

    # Try local import as fallback
    try:
        from speciesnet import (
            DEFAULT_MODEL,
            draw_bboxes,
            SpeciesNet,
            SUPPORTED_MODELS,
        )
    except ImportError as ie:
        print(f"Failed to import SpeciesNet: {ie}")
        # Cannot run without SpeciesNet, but the GUI should still show the error.
        SUPPORTED_MODELS = []
        DEFAULT_MODEL = "None"


# --- GLOBAL MODEL AND DEVICE INITIALIZATION ---
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    DEVICE_STATUS = f"CUDA is available. Running on GPU: {DEVICE}"
else:
    DEVICE = "cpu"
    DEVICE_STATUS = "CUDA not available. Running on CPU (performance will be slow)."

AVAILABLE_MODELS_INFO = f"Available models: {', '.join(SUPPORTED_MODELS) if 'SUPPORTED_MODELS' in locals() else 'N/A'}"
SELECTED_MODEL = DEFAULT_MODEL if "DEFAULT_MODEL" in locals() else "None"

model = None
MODEL_STATUS = "Initialization pending..."

try:
    if "SpeciesNet" in locals() and SELECTED_MODEL != "None":
        if KAGGLE_DOWNLOAD_SUCCESS:
            print(
                f"Initializing SpeciesNet with Kaggle Hub weights from: {KAGGLE_MODEL_PATH}"
            )
            model = SpeciesNet(SELECTED_MODEL)
            MODEL_STATUS = (
                f"✅ SpeciesNet model initialized with Kaggle Hub weights (v4.0.1b)"
            )
            MODEL_STATUS += f"\n📦 Model path: {KAGGLE_MODEL_PATH}"
        else:
            print(f"Initializing SpeciesNet with local weights")
            model = SpeciesNet(SELECTED_MODEL)
            MODEL_STATUS = f"SpeciesNet model initialized (local installation)"

        MODEL_STATUS += f"\n🎯 Selected model: {SELECTED_MODEL}"
        MODEL_STATUS += f"\n{AVAILABLE_MODELS_INFO}"

        # Manual GPU transfer fix
        if DEVICE.startswith("cuda"):
            if hasattr(model, "model") and model.model is not None:
                model.model.to(DEVICE)
                MODEL_STATUS += f"\n✅ Successfully moved internal model to {DEVICE}."
            elif hasattr(model, "yolo_model") and model.yolo_model is not None:
                model.yolo_model.to(DEVICE)
                MODEL_STATUS += (
                    f"\n✅ Successfully moved internal YOLO model to {DEVICE}."
                )
            else:
                MODEL_STATUS += (
                    "\n⚠️ Could not find internal model attribute. GPU transfer failed."
                )

        if not hasattr(model, "manager"):
            model.manager = None

        has_predict = hasattr(model, "predict") and callable(getattr(model, "predict"))
        has_classify = hasattr(model, "classify") and callable(
            getattr(model, "classify")
        )
        has_detect = hasattr(model, "detect") and callable(getattr(model, "detect"))

        MODEL_STATUS += f"\n✅ Available methods: predict={has_predict}, classify={has_classify}, detect={has_detect}"
    else:
        MODEL_STATUS = "ERROR: SpeciesNet import failed. Check installation."

except Exception as e:
    MODEL_STATUS = (
        f"ERROR during model initialization: {e}. Check SpeciesNet installation."
    )
    model = None

# --- HELPER FUNCTIONS (Copied from original script) ---


def format_classification_results(classifications: dict) -> str:
    """Format classification results into a readable string."""
    if not classifications:
        return "No classifications"

    # Handle SpeciesNet v4 format: {'classes': [...], 'scores': [...]}
    if (
        isinstance(classifications, dict)
        and "classes" in classifications
        and "scores" in classifications
    ):
        classes = classifications["classes"]
        scores = classifications["scores"]
        items = []
        for class_str, score in zip(classes, scores):
            parts = class_str.split(";")
            if len(parts) >= 7 and parts[6]:
                species_name = parts[6].strip()
            elif len(parts) >= 6 and parts[5]:
                genus = parts[4] if len(parts) > 4 else ""
                species = parts[5]
                species_name = f"{genus} {species}".strip() if genus else species
            elif len(parts) >= 2 and parts[1]:
                species_name = parts[1].strip()
            else:
                species_name = "Unknown"

            if species_name.lower() not in ["unknown", "blank", ""]:
                items.append((species_name.title(), float(score)))

        if not items:
            return "No valid species classifications"
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        results = [
            f"{i}. {species} ({confidence * 100:.1f}%)"
            for i, (species, confidence) in enumerate(sorted_items[:10], 1)
        ]
        return "\n".join(results)

    # Handle other formats (dict or list)
    items = []
    try:
        if isinstance(classifications, dict):
            for species, conf in classifications.items():
                if isinstance(conf, (int, float)) and not isinstance(
                    species, (int, float)
                ):
                    items.append((species, float(conf)))
        elif isinstance(classifications, list):
            for c in classifications:
                if isinstance(c, dict):
                    species = c.get("label", c.get("species", "Unknown"))
                    conf = c.get("confidence", 0.0)
                    if isinstance(conf, (int, float)) and isinstance(species, str):
                        items.append((species, float(conf)))
        else:
            return str(classifications)

        if not items:
            return "No valid classifications found"
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        results = [
            f"{i}. {species} ({confidence * 100:.1f}%)"
            for i, (species, confidence) in enumerate(sorted_items[:10], 1)
        ]
        return "\n".join(results)
    except Exception as e:
        return f"Error formatting classifications: {str(e)}"


def extract_species_label(detection: dict) -> tuple:
    """Extract the most specific species label and confidence from a detection."""
    label = detection.get("label", "Unknown")
    confidence = detection.get("confidence", 0.0)
    if "classification" in detection and isinstance(detection["classification"], dict):
        classification = detection["classification"]
        if classification:
            specific_label = max(classification, key=classification.get)
            specific_confidence = classification[specific_label]
            return f"{specific_label}", specific_confidence
    return label, confidence


def play_alarm_sound():
    """Play an alarm sound based on the operating system."""
    try:
        if platform.system() == "Windows":
            duration = 1000  # milliseconds
            freq = 1000  # Hz
            winsound.Beep(freq, duration)
        else:
            # For other OS, use a simple print statement as placeholder
            print("Alarm! Intruder detected!")
    except Exception as e:
        print(f"Error playing alarm sound: {e}")


# --- TKINTER APPLICATION CLASS ---


class SpeciesNetApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("SpeciesNet: Real-time Video Analysis with Intruder Alert")
        self.pack(fill="both", expand=True, padx=10, pady=10)

        # State variables
        self.video_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.max_frames = tk.IntVar(value=0)
        self.processing_mode = tk.StringVar(value="Detection + Classification")
        self.alarm_enabled = tk.BooleanVar(value=True)
        self.is_processing = False
        self.vidcap = None
        self.last_alarm_time = 0  # To prevent alarm spam
        self.alarm_cooldown = 3.0  # seconds between alarms

        # Create Widgets
        self.create_widgets()

        # Display initial status
        self.update_status_display()

    def create_widgets(self):
        # Configure grid for main frame
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # --- Status and Info Frame (Top Left) ---
        status_frame = ttk.LabelFrame(
            self, text="ℹ️ Application Status & Info", padding="10"
        )
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        self.status_text = scrolledtext.ScrolledText(
            status_frame, wrap=tk.WORD, height=12, width=80, font=("Courier", 9)
        )
        self.status_text.grid(row=0, column=0, sticky="ew")

        # --- Input/Controls Frame (Bottom Left) ---
        control_frame = ttk.LabelFrame(self, text="⚙️ Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        # Video Path
        ttk.Label(control_frame, text="Input Video:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Entry(control_frame, textvariable=self.video_path, width=40).grid(
            row=1, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(control_frame, text="Select Video", command=self.select_video).grid(
            row=1, column=1, sticky="ew"
        )

        # Output Directory
        ttk.Label(control_frame, text="Output Directory:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Entry(control_frame, textvariable=self.output_dir, width=40).grid(
            row=3, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(
            control_frame, text="Select Folder", command=self.select_output_dir
        ).grid(row=3, column=1, sticky="ew")

        # Max Frames
        ttk.Label(control_frame, text="Max Frames (0=Full Video):").grid(
            row=4, column=0, sticky="w", pady=2
        )
        ttk.Spinbox(
            control_frame,
            from_=0,
            to=100000,
            increment=1,
            textvariable=self.max_frames,
            width=10,
        ).grid(row=4, column=1, sticky="w")

        # Processing Mode
        ttk.Label(control_frame, text="Processing Mode:").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=5
        )
        for i, choice in enumerate(
            ["Detection + Classification", "Classification Only", "Detection Only"]
        ):
            ttk.Radiobutton(
                control_frame, text=choice, variable=self.processing_mode, value=choice
            ).grid(row=6 + i, column=0, columnspan=2, sticky="w")

        # Alarm Enable Checkbox
        ttk.Checkbutton(
            control_frame,
            text="🔊 Enable Alarm Sound for Intruder Detection",
            variable=self.alarm_enabled,
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=5)

        # Start/Stop Button
        self.process_button = ttk.Button(
            control_frame, text="▶️ Start Processing", command=self.start_processing
        )
        self.process_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=10)

        # --- Output/Result Frame (Bottom Right) ---
        output_frame = ttk.LabelFrame(self, text="📊 Results", padding="10")
        output_frame.grid(row=1, column=1, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)

        # Alert Label (NEW)
        self.alert_label = ttk.Label(
            output_frame, text="", font=("Arial", 14, "bold"), foreground="red"
        )
        self.alert_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        # Image Output
        self.image_label = ttk.Label(
            output_frame, text="Annotated Frame Will Appear Here"
        )
        self.image_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # FPS
        ttk.Label(output_frame, text="Processing Speed (FPS):").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.fps_var = tk.StringVar(value="Waiting...")
        ttk.Label(
            output_frame, textvariable=self.fps_var, font=("Arial", 10, "bold")
        ).grid(row=2, column=1, sticky="e", pady=2)

        # Detections
        ttk.Label(output_frame, text="Object Detections:").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2
        )
        self.detections_text = scrolledtext.ScrolledText(
            output_frame, wrap=tk.WORD, height=3, width=40
        )
        self.detections_text.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

        # Classifications
        ttk.Label(output_frame, text="Species Classifications (Top 10):").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=2
        )
        self.classifications_text = scrolledtext.ScrolledText(
            output_frame, wrap=tk.WORD, height=8, width=40
        )
        self.classifications_text.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=2
        )

        # Initial empty image to avoid layout issues
        self.empty_img = ImageTk.PhotoImage(
            Image.new("RGB", (400, 300), color="lightgray")
        )
        self.image_label.config(image=self.empty_img)

    def update_status_display(self):
        """Populates the ScrolledText widget with the initial status HTML converted to text."""

        # Simple conversion of the Gradio HTML to a readable string for Tkinter
        status_lines = [
            "🔬 SpeciesNet Video Analyzer - Vidarbha Region with Intruder Alert System",
            f"Device Status: {DEVICE_STATUS}",
            f"Model Status: {MODEL_STATUS.replace('\n', ' ')}",
            f"Model Source: {'✅ Kaggle Hub' if KAGGLE_DOWNLOAD_SUCCESS else '⚠️ Local Installation'}",
            "🌍 Geofencing: Configured for Vidarbha, Maharashtra, India (21.1458°N, 79.0882°E)",
            "🚨 Alarm: Triggers when objects/animals are detected in the video",
            "--------------------------------------------------",
            "📋 Processing Modes:",
            " - Detection + Classification: Full pipeline - locates animals with bounding boxes AND classifies species",
            " - Classification Only: Identifies species in the entire frame (no bounding boxes, faster)",
            " - Detection Only: Locates animals with bounding boxes only (no species classification)",
            "--------------------------------------------------",
            "🐾 Common Wildlife in Vidarbha Region (Geofencing Hint):",
            "Mammals: Bengal Tiger, Indian Leopard, Sloth Bear, Wild Boar, Spotted Deer (Chital), Sambar Deer, Nilgai (Blue Bull), Indian Gaur (Bison), Striped Hyena, Indian Wolf, Golden Jackal, Indian Fox, Mongoose, Langur, Rhesus Macaque",
            "Birds: Peacock (Indian Peafowl), Red Junglefowl, Various Eagles and Vultures, Owls",
            "Reptiles: Indian Python, Cobra, Monitor Lizard, Crocodile (in water bodies)",
            "--------------------------------------------------",
            "💡 Note: The model uses geofencing to prioritize species known to occur in the Vidarbha region.",
        ]

        status_content = "\n".join(status_lines)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, status_content)
        self.status_text.config(state=tk.DISABLED)  # Make it read-only

    def select_video(self):
        filetypes = (("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(
            title="Select Input Video File", filetypes=filetypes
        )
        if filepath:
            self.video_path.set(filepath)
            # Set default output dir to the video's directory
            self.output_dir.set(os.path.dirname(filepath))

    def select_output_dir(self):
        directory = filedialog.askdirectory(
            title="Select Output Directory for Processed Video"
        )
        if directory:
            self.output_dir.set(directory)

    def trigger_alarm(self):
        """Trigger alarm sound and visual alert."""
        current_time = time.time()
        if current_time - self.last_alarm_time >= self.alarm_cooldown:
            self.last_alarm_time = current_time

            # Visual alert
            self.master.after(0, self.show_alert_message, "⚠️ INTRUDER DETECTED!")

            # Play sound in separate thread to avoid blocking
            if self.alarm_enabled.get():
                alarm_thread = threading.Thread(target=play_alarm_sound, daemon=True)
                alarm_thread.start()

    def show_alert_message(self, message):
        """Show alert message in the UI."""
        self.alert_label.config(text=message)
        # Clear alert after 2 seconds
        self.master.after(2000, lambda: self.alert_label.config(text=""))

    def start_processing(self):
        if self.is_processing:
            # This would be the 'Stop' functionality
            self.is_processing = False
            self.process_button.config(text="Stopping...", state=tk.DISABLED)
            return

        video_path = self.video_path.get()
        output_dir = self.output_dir.get()
        max_frames = self.max_frames.get()
        mode = self.processing_mode.get()

        if model is None:
            messagebox.showerror(
                "Error", "Model failed to initialize. Cannot run prediction."
            )
            return

        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid input video file.")
            return

        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return

        self.is_processing = True
        self.process_button.config(
            text="⏹️ Stop Processing", command=self.stop_processing
        )

        # Clear output fields
        self.fps_var.set("Processing...")
        self.alert_label.config(text="")
        self.detections_text.config(state=tk.NORMAL)
        self.detections_text.delete(1.0, tk.END)
        self.detections_text.insert(tk.END, "Analyzing video...")
        self.detections_text.config(state=tk.DISABLED)
        self.classifications_text.config(state=tk.NORMAL)
        self.classifications_text.delete(1.0, tk.END)
        self.classifications_text.insert(tk.END, "Analyzing video...")
        self.classifications_text.config(state=tk.DISABLED)

        # Start the heavy processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self.process_video_thread,
            args=(video_path, max_frames, mode, output_dir),
            daemon=True,
        )
        self.processing_thread.start()

    def stop_processing(self):
        self.is_processing = False
        self.process_button.config(text="Stopping...", state=tk.DISABLED)
        if self.vidcap:
            self.vidcap.release()
            self.vidcap = None

    def update_ui_from_thread(
        self,
        annotated_img,
        current_fps,
        species_summary,
        classification_summary,
        has_detections,
    ):
        """Safely updates Tkinter widgets from a worker thread using master.after."""
        if not self.is_processing:
            return

        # Trigger alarm if detections found
        if has_detections:
            self.trigger_alarm()

        # 1. Update Image
        img_w, img_h = annotated_img.size
        # Resize the image to fit the label, e.g., max width 600
        max_w = 600
        if img_w > max_w:
            ratio = max_w / img_w
            new_w = max_w
            new_h = int(img_h * ratio)
            annotated_img = annotated_img.resize(
                (new_w, new_h), Image.Resampling.LANCZOS
            )

        self.tk_img = ImageTk.PhotoImage(annotated_img)
        self.image_label.config(image=self.tk_img, text="")
        self.image_label.image = self.tk_img

        # 2. Update FPS
        self.fps_var.set(current_fps)

        # 3. Update Detections
        self.detections_text.config(state=tk.NORMAL)
        self.detections_text.delete(1.0, tk.END)
        self.detections_text.insert(tk.END, species_summary)
        self.detections_text.config(state=tk.DISABLED)

        # 4. Update Classifications
        self.classifications_text.config(state=tk.NORMAL)
        self.classifications_text.delete(1.0, tk.END)
        self.classifications_text.insert(tk.END, classification_summary)
        self.classifications_text.config(state=tk.DISABLED)

    def finalize_processing(self, final_fps, output_filepath):
        """Finalizes the UI after processing is complete or stopped."""
        self.is_processing = False
        self.process_button.config(text="▶️ Start Processing", state=tk.NORMAL)
        self.fps_var.set(final_fps)
        self.alert_label.config(text="")

        # Update text outputs with final status
        self.detections_text.config(state=tk.NORMAL)
        self.detections_text.delete(1.0, tk.END)
        self.detections_text.insert(
            tk.END, f"Stream Complete. Output saved to:\n{output_filepath}"
        )
        self.detections_text.config(state=tk.DISABLED)

        self.classifications_text.config(state=tk.NORMAL)
        self.classifications_text.delete(1.0, tk.END)
        self.classifications_text.insert(tk.END, "Processing complete.")
        self.classifications_text.config(state=tk.DISABLED)

    # --- CORE VIDEO PROCESSING FUNCTION (Modified for threading) ---
    def process_video_thread(
        self, video_path: str, max_frames: int, mode: str, output_dir: str
    ):
        """
        Processes an uploaded video file frame-by-frame. Runs in a separate thread.

        Args:
            video_path: Path to the video file.
            max_frames: Max frames to process (0 for full video).
            mode: Processing mode.
            output_dir: Directory to save the output video.
        """

        # Determine processing mode
        run_full_predict = mode == "Detection + Classification"
        run_classify_only = mode == "Classification Only"
        run_detect_only = mode == "Detection Only"

        temp_frame_path = None
        output_filepath = os.path.join(
            output_dir, f"speciesnet_output_{os.path.basename(video_path)}"
        )
        video_writer = None

        try:
            # Setup Video I/O
            self.vidcap = cv2.VideoCapture(video_path)
            if not self.vidcap.isOpened():
                self.master.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Could not open video file {video_path}"
                    ),
                )
                return

            # Get video properties
            fps = self.vidcap.get(cv2.CAP_PROP_FPS)
            width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Setup VideoWriter
            video_writer = cv2.VideoWriter(
                output_filepath, fourcc, fps, (width, height)
            )
            if not video_writer.isOpened():
                self.master.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Could not open VideoWriter at {output_filepath}"
                    ),
                )
                return

            # Setup Temp File
            temp_dir = tempfile.gettempdir()
            temp_frame_path = os.path.join(
                temp_dir, f"speciesnet_frame_{os.getpid()}_{time.time()}.jpg"
            )

            frame_count = 0
            start_time = time.time()

            while self.is_processing and self.vidcap.isOpened():
                if max_frames > 0 and frame_count >= max_frames:
                    break

                success, frame = self.vidcap.read()
                if not success:
                    break

                frame_start_time = time.time()

                # Convert to PIL RGB Image
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_img.save(temp_frame_path, format="JPEG", quality=95)

                # Initialize results
                detections = []
                species_summary = "N/A"
                classification_summary = "N/A"
                annotated_img = pil_img
                has_detections = False

                try:
                    if run_full_predict:
                        predictions_dict = model.predict(
                            instances_dict={
                                "instances": [
                                    {
                                        "filepath": temp_frame_path,
                                        "latitude": 21.1458,
                                        "longitude": 79.0882,
                                        "country": "IND",
                                    }
                                ]
                            }
                        )
                        if predictions_dict and "predictions" in predictions_dict:
                            pred = predictions_dict["predictions"][0]
                            detections = pred.get("detections", [])
                            if detections:
                                has_detections = True
                                detailed_labels = [
                                    f"{extract_species_label(det)[0]} ({extract_species_label(det)[1] * 100:.1f}%)"
                                    for det in detections
                                ]
                                species_summary = " | ".join(detailed_labels)
                                annotated_img = draw_bboxes(pil_img, detections)
                            else:
                                species_summary = "No detections"

                            classifications = pred.get("classification") or pred.get(
                                "classifications"
                            )
                            if classifications:
                                classification_summary = format_classification_results(
                                    classifications
                                )
                            else:
                                all_species = {}
                                for det in detections:
                                    if "classification" in det and isinstance(
                                        det["classification"], dict
                                    ):
                                        for species, conf in det[
                                            "classification"
                                        ].items():
                                            if isinstance(species, str) and isinstance(
                                                conf, (int, float)
                                            ):
                                                all_species[species] = max(
                                                    all_species.get(species, 0.0),
                                                    float(conf),
                                                )
                                if all_species:
                                    classification_summary = (
                                        format_classification_results(all_species)
                                    )
                                else:
                                    classification_summary = (
                                        "No detailed classification"
                                    )

                    elif run_classify_only:
                        classify_dict = model.classify(filepaths=[temp_frame_path])
                        if classify_dict and "predictions" in classify_dict:
                            pred = classify_dict["predictions"][0]
                            classifications = (
                                pred.get("classifications")
                                or pred.get("classification")
                                or {
                                    k: v
                                    for k, v in pred.items()
                                    if isinstance(v, (int, float))
                                    and k
                                    not in [
                                        "filepath",
                                        "latitude",
                                        "longitude",
                                        "country",
                                    ]
                                }
                            )
                            if classifications:
                                has_detections = (
                                    True  # Trigger alarm for classifications
                                )
                                classification_summary = format_classification_results(
                                    classifications
                                )
                            else:
                                classification_summary = "No results from classifier"
                            species_summary = "Classification mode (no bounding boxes)"

                    elif run_detect_only:
                        detect_dict = model.detect(filepaths=[temp_frame_path])
                        if detect_dict and "predictions" in detect_dict:
                            pred = detect_dict["predictions"][0]
                            detections = pred.get("detections", [])
                            if detections:
                                has_detections = True
                                detailed_labels = [
                                    f"{det.get('label', 'Unknown')} ({det.get('confidence', 0.0) * 100:.1f}%)"
                                    for det in detections
                                ]
                                species_summary = " | ".join(detailed_labels)
                                annotated_img = draw_bboxes(pil_img, detections)
                            else:
                                species_summary = "No detections"
                            classification_summary = (
                                "Detection-only mode (no species classification)"
                            )
                        else:
                            species_summary = "No results from detector"

                except Exception as e:
                    error_msg = f"⚠️ Error on frame {frame_count}: {str(e)}"
                    print(error_msg)
                    species_summary = error_msg
                    classification_summary = error_msg

                finally:
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)

                # Write annotated frame to video file
                annotated_np = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
                video_writer.write(annotated_np)

                # Calculate FPS
                frame_processing_time = time.time() - frame_start_time
                current_fps = f"{1.0 / frame_processing_time:.2f} FPS"

                # Safely update the GUI in the main thread
                self.master.after(
                    0,
                    self.update_ui_from_thread,
                    annotated_img,
                    current_fps,
                    species_summary,
                    classification_summary,
                    has_detections,
                )

                frame_count += 1

        except Exception as e:
            self.master.after(
                0,
                lambda: messagebox.showerror(
                    "Processing Error", f"A fatal error occurred: {str(e)}"
                ),
            )
            print(f"FATAL ERROR: {e}")

        finally:
            if self.vidcap is not None:
                self.vidcap.release()
                self.vidcap = None
            if video_writer is not None:
                video_writer.release()
            if temp_frame_path and os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

        # Final cleanup and status update
        total_time = time.time() - start_time
        final_fps = (
            f"{frame_count / total_time:.2f} FPS (Avg)"
            if total_time > 0 and frame_count > 0
            else "0.00 FPS"
        )

        self.master.after(0, self.finalize_processing, final_fps, output_filepath)


# --- MAIN APPLICATION LAUNCH ---
if __name__ == "__main__":
    # Ensure draw_bboxes is available if SpeciesNet loaded successfully
    if "draw_bboxes" not in locals():

        def draw_bboxes(img, detections):
            print("Warning: draw_bboxes placeholder used.")
            return img

    root = tk.Tk()
    app = SpeciesNetApp(master=root)
    root.mainloop()
