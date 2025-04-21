import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import torch
import torchaudio
import subprocess
from PIL import Image, ImageTk

class DeepfakeGenerator:
    """Core deepfake generation engine that handles video, audio and face manipulation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.progress_callback = None
        self.stop_requested = False
        
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def request_stop(self):
        """Request to stop processing"""
        self.stop_requested = True
        
    def reset(self):
        """Reset the generator state"""
        self.stop_requested = False
        
    def update_progress(self, progress, message=""):
        """Update progress through callback"""
        if self.progress_callback:
            self.progress_callback(progress, message)
            
    def load_face_model(self):
        """Load pre-trained face manipulation model"""
        self.update_progress(5, "Loading face manipulation model...")
        # Simulating model loading
        time.sleep(1)
        return True
        
    def load_voice_model(self):
        """Load pre-trained voice synthesis model"""
        self.update_progress(10, "Loading voice synthesis model...")
        # Simulating model loading
        time.sleep(1)
        return True
        
    def extract_face_features(self, video_path):
        """Extract facial features from the source video"""
        self.update_progress(15, "Extracting facial features...")
        
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Simulate feature extraction
        features = {"landmarks": [], "expressions": [], "identity": None}
        
        # Process frames to extract facial features
        for i in range(min(100, frame_count)):
            if self.stop_requested:
                cap.release()
                return None
                
            ret, frame = cap.read()
            if not ret:
                break
                
            # Simulate face detection and feature extraction
            # In a real implementation, this would use a face detection library
            time.sleep(0.05)
            
            # Update progress
            progress = 15 + (i / min(100, frame_count)) * 15
            self.update_progress(progress, f"Processing frame {i+1}/{min(100, frame_count)}")
            
        cap.release()
        self.update_progress(30, "Facial features extracted")
        return features
        
    def extract_voice(self, video_path):
        """Extract voice characteristics from the source video"""
        self.update_progress(35, "Extracting voice characteristics...")
        
        # Simulate voice extraction
        time.sleep(2)
        
        voice_features = {
            "pitch_range": (80, 180),
            "timbre": "warm",
            "speech_rate": 1.0
        }
        
        self.update_progress(40, "Voice characteristics extracted")
        return voice_features
        
    def generate_synthetic_voice(self, text, voice_features, output_audio_path, language="en"):
        """Generate synthetic voice saying the provided text"""
        self.update_progress(45, "Generating synthetic voice...")
        
        # Simulate voice synthesis
        for i in range(10):
            if self.stop_requested:
                return False
                
            time.sleep(0.2)
            progress = 45 + (i / 10) * 15
            self.update_progress(progress, f"Synthesizing voice: {i*10}%")
            
        # Create a dummy audio file
        dummy_audio = np.random.uniform(-0.1, 0.1, size=16000*5)  # 5 seconds of audio
        dummy_audio = (dummy_audio * 32767).astype(np.int16)
        
        # Save the audio
        with open(output_audio_path, 'wb') as f:
            f.write(b'RIFF')
            f.write((36 + len(dummy_audio)*2).to_bytes(4, 'little'))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))
            f.write((1).to_bytes(2, 'little'))  # PCM
            f.write((1).to_bytes(2, 'little'))  # Mono
            f.write((16000).to_bytes(4, 'little'))  # Sample rate
            f.write((16000*2).to_bytes(4, 'little'))  # Byte rate
            f.write((2).to_bytes(2, 'little'))  # Block align
            f.write((16).to_bytes(2, 'little'))  # Bits per sample
            f.write(b'data')
            f.write((len(dummy_audio)*2).to_bytes(4, 'little'))
            f.write(dummy_audio.tobytes())
        
        self.update_progress(60, "Synthetic voice generated")
        return True
        
    def generate_deepfake_video(self, source_video, face_features, text_to_say, output_audio_path, output_video_path):
        """Generate deepfake video with the person saying the provided text"""
        self.update_progress(65, "Generating deepfake video...")
        
        # Check if file exists
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Source video file not found: {source_video}")
            
        # Open the video file
        source_cap = cv2.VideoCapture(source_video)
        if not source_cap.isOpened():
            raise ValueError(f"Could not open source video: {source_video}")
            
        # Get video properties
        width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = source_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Simulate video generation process
        total_frames = 150  # Generate about 5 seconds of video at 30fps
        
        for i in range(total_frames):
            if self.stop_requested:
                source_cap.release()
                out.release()
                return False
                
            # Simulate frame processing
            ret, frame = source_cap.read()
            if not ret:
                # If we reach the end of the source video, loop back
                source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = source_cap.read()
                
            # Simulate deepfake frame generation
            time.sleep(0.05)
            
            # Draw some text to show this is a simulated output
            cv2.putText(frame, f"Deepfake: {text_to_say[:20]}...", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Write the frame to output video
            out.write(frame)
            
            # Update progress
            progress = 65 + (i / total_frames) * 25
            self.update_progress(progress, f"Generating frame {i+1}/{total_frames}")
            
        source_cap.release()
        out.release()
        
        # Combine audio and video
        self.update_progress(90, "Combining audio and video...")
        self.combine_audio_video(output_video_path, output_audio_path, output_video_path.replace('.mp4', '_final.mp4'))
        
        self.update_progress(100, "Deepfake video generated successfully!")
        return output_video_path.replace('.mp4', '_final.mp4')
        
    def combine_audio_video(self, video_path, audio_path, output_path):
        """Combine video and audio using FFmpeg or similar"""
        # In a real implementation, this would use FFmpeg 
        # For simulation, just copy the video file
        if os.path.exists(video_path):
            with open(video_path, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
        return output_path
        
    def process(self, source_video, text_to_say, output_path, language="en"):
        """Main processing pipeline"""
        try:
            # Create temporary directory for intermediate files
            temp_dir = os.path.join(os.path.dirname(output_path), "temp_deepfake")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Load models
            self.load_face_model()
            self.load_voice_model()
            
            # Extract features
            face_features = self.extract_face_features(source_video)
            if self.stop_requested or face_features is None:
                return False
                
            voice_features = self.extract_voice(source_video)
            if self.stop_requested:
                return False
                
            # Generate audio
            audio_output = os.path.join(temp_dir, "synthetic_voice.wav")
            success = self.generate_synthetic_voice(text_to_say, voice_features, audio_output, language)
            if not success or self.stop_requested:
                return False
                
            # Generate video
            video_output = os.path.join(temp_dir, "deepfake_raw.mp4")
            final_output = self.generate_deepfake_video(source_video, face_features, 
                                                       text_to_say, audio_output, video_output)
            
            # Cleanup temp files (in a real implementation)
            # shutil.rmtree(temp_dir)
            
            return final_output
            
        except Exception as e:
            self.update_progress(-1, f"Error: {str(e)}")
            print(f"Error in processing: {str(e)}")
            return False


class DeepfakeApp(tk.Tk):
    """Main application window for the Deepfake Generator"""
    
    def __init__(self):
        super().__init__()
        
        self.title("AI Deepfake Generator")
        self.geometry("800x600")
        self.minsize(800, 600)
        
        # Set icon if available
        try:
            self.iconbitmap("icon.ico")
        except:
            pass
            
        # Initialize the deepfake generator
        self.generator = DeepfakeGenerator()
        self.generator.set_progress_callback(self.update_progress)
        
        # State variables
        self.source_video = tk.StringVar()
        self.output_path = tk.StringVar()
        self.text_to_say = tk.StringVar()
        self.language = tk.StringVar(value="en")
        self.progress_var = tk.DoubleVar()
        self.status_text = tk.StringVar(value="Ready")
        self.processing = False
        self.output_video_path = None
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all UI elements"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Settings")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Source video selection
        ttk.Label(input_frame, text="Source Video:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.source_video, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_source_video).grid(row=0, column=2, padx=5, pady=5)
        
        # Output path selection
        ttk.Label(input_frame, text="Output Location:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_output_path).grid(row=1, column=2, padx=5, pady=5)
        
        # Text to say
        ttk.Label(input_frame, text="Text to Say:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        text_entry = ttk.Entry(input_frame, textvariable=self.text_to_say, width=50)
        text_entry.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # Language selection
        ttk.Label(input_frame, text="Language:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        languages = ["en", "es", "fr", "de", "it", "ja", "ko", "zh", "ru"]
        language_dropdown = ttk.Combobox(input_frame, textvariable=self.language, values=languages, width=5)
        language_dropdown.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Process control section
        process_frame = ttk.LabelFrame(main_frame, text="Process Control")
        process_frame.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        ttk.Label(process_frame, textvariable=self.status_text).pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(process_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Process", command=self.start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Process", command=self.stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Output playback section
        playback_frame = ttk.LabelFrame(main_frame, text="Output Video Playback")
        playback_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_canvas = tk.Canvas(playback_frame, bg="black", width=640, height=360)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Playback buttons
        playback_buttons_frame = ttk.Frame(playback_frame)
        playback_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Row 1 of playback buttons
        ttk.Button(playback_buttons_frame, text="VLC Player", 
                  command=lambda: self.play_video("vlc")).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="Windows Media Player", 
                  command=lambda: self.play_video("wmplayer")).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="QuickTime Player", 
                  command=lambda: self.play_video("quicktime")).grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="MPC Player", 
                  command=lambda: self.play_video("mpc")).grid(row=0, column=3, padx=5, pady=2)
        
        # Row 2 of playback buttons
        ttk.Button(playback_buttons_frame, text="PotPlayer", 
                  command=lambda: self.play_video("potplayer")).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="KMPlayer", 
                  command=lambda: self.play_video("kmplayer")).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="GOM Player", 
                  command=lambda: self.play_video("gomplayer")).grid(row=1, column=2, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="VLC with Subtitles", 
                  command=lambda: self.play_video("vlc", subtitles=True)).grid(row=1, column=3, padx=5, pady=2)
        
        # Row 3 of playback buttons
        ttk.Button(playback_buttons_frame, text="VLC with Audio Description", 
                  command=lambda: self.play_video("vlc", audio_description=True)).grid(row=2, column=0, columnspan=2, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="VLC with Audio Description & Subtitles", 
                  command=lambda: self.play_video("vlc", subtitles=True, audio_description=True)).grid(row=2, column=2, columnspan=2, padx=5, pady=2)
        
        # Exit button
        ttk.Button(main_frame, text="Exit", command=self.quit).pack(side=tk.RIGHT, pady=10)
        
    def browse_source_video(self):
        """Open a file dialog to choose the source video"""
        filetypes = [
            ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(title="Select Source Video", filetypes=filetypes)
        if filename:
            self.source_video.set(filename)
            
    def browse_output_path(self):
        """Open a file dialog to choose the output path"""
        filetypes = [("MP4 files", "*.mp4")]
        filename = filedialog.asksaveasfilename(title="Select Output Location", 
                                               defaultextension=".mp4", filetypes=filetypes)
        if filename:
            self.output_path.set(filename)
            
    def validate_inputs(self):
        """Validate all inputs before starting the process"""
        if not self.source_video.get():
            messagebox.showerror("Error", "Please select a source video")
            return False
            
        if not os.path.exists(self.source_video.get()):
            messagebox.showerror("Error", "Source video file does not exist")
            return False
            
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output location")
            return False
            
        if not self.text_to_say.get():
            messagebox.showerror("Error", "Please enter text for the deepfake to say")
            return False
            
        return True
        
    def start_process(self):
        """Start the deepfake generation process"""
        if not self.validate_inputs():
            return
            
        if self.processing:
            messagebox.showinfo("Info", "Process already running")
            return
            
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start processing in a separate thread
        threading.Thread(target=self.run_processing, daemon=True).start()
        
    def run_processing(self):
        """Run the processing in a separate thread"""
        try:
            # Reset progress
            self.progress_var.set(0)
            self.status_text.set("Starting process...")
            
            # Reset generator
            self.generator.reset()
            
            # Start processing
            result = self.generator.process(
                self.source_video.get(),
                self.text_to_say.get(),
                self.output_path.get(),
                self.language.get()
            )
            
            # Handle result
            if result:
                self.output_video_path = result
                self.status_text.set("Process completed successfully!")
                self.show_preview_frame()
                
                # Show success message
                messagebox.showinfo("Success", "Deepfake generated successfully!")
            else:
                if self.generator.stop_requested:
                    self.status_text.set("Process stopped by user")
                else:
                    self.status_text.set("Process failed. See console for details.")
                    
        except Exception as e:
            self.status_text.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
        finally:
            # Reset UI state
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
    def stop_process(self):
        """Stop the running process"""
        if self.processing:
            self.generator.request_stop()
            self.status_text.set("Stopping process...")
            
    def update_progress(self, progress, message=""):
        """Update the progress bar and status text"""
        self.progress_var.set(progress)
        if message:
            self.status_text.set(message)
        
        # Update UI
        self.update_idletasks()
        
    def show_preview_frame(self):
        """Show a preview frame from the generated video"""
        if not self.output_video_path or not os.path.exists(self.output_video_path):
            return
            
        # Try to capture a frame from the video
        try:
            cap = cv2.VideoCapture(self.output_video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to fit the canvas
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    
                    if canvas_width / canvas_height > aspect_ratio:
                        # Canvas is wider than video
                        height = canvas_height
                        width = int(height * aspect_ratio)
                    else:
                        # Canvas is taller than video
                        width = canvas_width
                        height = int(width / aspect_ratio)
                        
                    frame = cv2.resize(frame, (width, height))
                    
                    # Convert to PhotoImage
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    # Update canvas
                    self.preview_canvas.create_image(
                        canvas_width//2, canvas_height//2,
                        image=photo, anchor=tk.CENTER
                    )
                    
                    # Keep a reference to prevent garbage collection
                    self.preview_canvas.image = photo
        except Exception as e:
            print(f"Error showing preview: {str(e)}")
            
    def play_video(self, player_name, subtitles=False, audio_description=False):
        """Play the generated video in the selected player"""
        if not self.output_video_path or not os.path.exists(self.output_video_path):
            messagebox.showerror("Error", "No output video available")
            return
            
        try:
            if player_name == "vlc":
                args = ["vlc"]
                if subtitles:
                    args.append("--sub-file=subtitles.srt")  # Would need to generate subtitles
                if audio_description:
                    args.append("--audio-track=2")  # Would need to generate audio description
                args.append(self.output_video_path)
                subprocess.Popen(args)
                
            elif player_name == "wmplayer":
                subprocess.Popen(["wmplayer", self.output_video_path])
                
            elif player_name == "quicktime":
                subprocess.Popen(["open", "-a", "QuickTime Player", self.output_video_path])
                
            elif player_name == "mpc":
                subprocess.Popen(["mpc-hc", self.output_video_path])
                
            elif player_name == "potplayer":
                subprocess.Popen(["potplayer", self.output_video_path])
                
            elif player_name == "kmplayer":
                subprocess.Popen(["kmplayer", self.output_video_path])
                
            elif player_name == "gomplayer":
                subprocess.Popen(["gomplayer", self.output_video_path])
                
            else:
                # Default to system default player
                if sys.platform == "win32" or sys.platform == "win64":
                    output_folder = os.path.dirname(os.path.abspath(__file__))
                    output_video_path = os.path.join(output_folder, os.path.basename(self.output_video_path))
                    os.rename(self.output_video_path, output_video_path)
                    os.startfile(output_video_path)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", self.output_video_path])
                else:
                    subprocess.Popen(["xdg-open", self.output_video_path])
                    
        except Exception as e:
            messagebox.showerror("Error", f"Could not play video: {str(e)}")
            

if __name__ == "__main__":
    # Check for required libraries
    try:
        import cv2
        import numpy as np
        import torch
        import torchaudio
    except ImportError as e:
        print(f"Required library not found: {str(e)}")
        print("Please install the missing libraries with: pip install opencv-python numpy torch torchaudio")
        sys.exit(1)
        
    # Start the application
    app = DeepfakeApp()
    app.mainloop()