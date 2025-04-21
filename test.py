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
import shutil
from PIL import Image, ImageTk

# Additional imports for deepfake functionality
import torch.nn as nn
import torch.nn.functional as F
import face_recognition
import librosa
import soundfile as sf
import tempfile
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip


class FaceSwapModel(nn.Module):
    """Neural network for face swapping"""
    
    def __init__(self):
        super(FaceSwapModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VoiceCloneModel(nn.Module):
    """Neural network for voice cloning"""
    
    def __init__(self):
        super(VoiceCloneModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=4, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=4, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=15, stride=4, padding=7, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(1, 2)  # Reshape for LSTM
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)  # Reshape back for ConvTranspose
        x = self.decoder(x)
        return x


class LipSyncModel(nn.Module):
    """Neural network for lip syncing"""
    
    def __init__(self):
        super(LipSyncModel, self).__init__()
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Face encoder
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Fusion and generation
        self.fusion = nn.Sequential(
            nn.Conv2d(128+128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, face, audio):
        face_features = self.face_encoder(face)
        audio_features = self.audio_encoder(audio)
        
        # Reshape audio features to match face dimensions
        batch_size, channels, _ = audio_features.shape
        h, w = face_features.shape[2], face_features.shape[3]
        audio_features = audio_features.view(batch_size, channels, 1, -1)
        audio_features = F.interpolate(audio_features, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate features
        combined = torch.cat([face_features, audio_features], dim=1)
        output = self.fusion(combined)
        return output


class DeepfakeGenerator:
    """Core deepfake generation engine that handles video, audio and face manipulation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.progress_callback = None
        self.stop_requested = False
        
        # Models
        self.face_swap_model = None
        self.voice_clone_model = None
        self.lip_sync_model = None
        
        # Pre-trained weights location
        self.weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        
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
            
    def download_models(self):
        """Download pre-trained models if not available locally"""
        self.update_progress(2, "Checking pre-trained models...")
        
        # Check for face swap model
        face_swap_path = os.path.join(self.weights_dir, "face_swap_model.pt")
        if not os.path.exists(face_swap_path):
            self.update_progress(2, "Downloading face swap model...")
            # In a real implementation, this would download from a server
            # For this example, we'll simulate the download
            time.sleep(1)
            # Create a dummy model file
            torch.save({"state_dict": FaceSwapModel().state_dict()}, face_swap_path)
        
        # Check for voice clone model
        voice_clone_path = os.path.join(self.weights_dir, "voice_clone_model.pt")
        if not os.path.exists(voice_clone_path):
            self.update_progress(3, "Downloading voice clone model...")
            time.sleep(1)
            torch.save({"state_dict": VoiceCloneModel().state_dict()}, voice_clone_path)
        
        # Check for lip sync model
        lip_sync_path = os.path.join(self.weights_dir, "lip_sync_model.pt")
        if not os.path.exists(lip_sync_path):
            self.update_progress(4, "Downloading lip sync model...")
            time.sleep(1)
            torch.save({"state_dict": LipSyncModel().state_dict()}, lip_sync_path)
            
        self.update_progress(5, "All models available")
        return True
    
    def load_face_model(self):
        """Load pre-trained face manipulation model"""
        self.update_progress(6, "Loading face swap model...")
        
        try:
            # Initialize the model
            self.face_swap_model = FaceSwapModel().to(self.device)
            
            # Load pre-trained weights
            checkpoint = torch.load(os.path.join(self.weights_dir, "face_swap_model.pt"), 
                                    map_location=self.device)
            self.face_swap_model.load_state_dict(checkpoint["state_dict"])
            
            # Set to evaluation mode
            self.face_swap_model.eval()
            self.update_progress(8, "Face swap model loaded")
            return True
        except Exception as e:
            self.update_progress(-1, f"Error loading face model: {str(e)}")
            return False
            
    def load_voice_model(self):
        """Load pre-trained voice synthesis model"""
        self.update_progress(9, "Loading voice clone model...")
        
        try:
            # Initialize the model
            self.voice_clone_model = VoiceCloneModel().to(self.device)
            
            # Load pre-trained weights
            checkpoint = torch.load(os.path.join(self.weights_dir, "voice_clone_model.pt"), 
                                   map_location=self.device)
            self.voice_clone_model.load_state_dict(checkpoint["state_dict"])
            
            # Set to evaluation mode
            self.voice_clone_model.eval()
            
            # Load lip sync model
            self.update_progress(10, "Loading lip sync model...")
            self.lip_sync_model = LipSyncModel().to(self.device)
            
            # Load pre-trained weights
            checkpoint = torch.load(os.path.join(self.weights_dir, "lip_sync_model.pt"), 
                                   map_location=self.device)
            self.lip_sync_model.load_state_dict(checkpoint["state_dict"])
            
            # Set to evaluation mode
            self.lip_sync_model.eval()
            
            self.update_progress(12, "Voice and lip sync models loaded")
            return True
        except Exception as e:
            self.update_progress(-1, f"Error loading voice model: {str(e)}")
            return False
        
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
        
        # Features to extract
        features = {
            "landmarks": [],
            "face_images": [],
            "face_encodings": [],
            "frames": []
        }
        
        # Process frames to extract facial features
        sample_rate = max(1, frame_count // 100)  # Sample frames for efficiency
        frame_idx = 0
        
        while True:
            if self.stop_requested:
                cap.release()
                return None
                
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate == 0:
                # Convert to RGB (face_recognition uses RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    # Get facial landmarks
                    landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    if landmarks and face_encodings:
                        # Extract the face region
                        top, right, bottom, left = face_locations[0]
                        face_image = rgb_frame[top:bottom, left:right]
                        
                        # Resize for consistency
                        if face_image.size > 0:
                            face_image = cv2.resize(face_image, (256, 256))
                            
                            # Store data
                            features["landmarks"].append(landmarks[0])
                            features["face_images"].append(face_image)
                            features["face_encodings"].append(face_encodings[0])
                            features["frames"].append(frame_idx)
                
                # Update progress
                progress = 15 + (frame_idx / frame_count) * 10
                self.update_progress(progress, f"Processing frame {frame_idx}/{frame_count}")
                
            frame_idx += 1
            
        cap.release()
        
        if not features["landmarks"]:
            raise ValueError("No faces detected in the video")
            
        self.update_progress(25, f"Extracted {len(features['landmarks'])} facial feature sets")
        return features
        
    def extract_voice(self, video_path, temp_dir):
        """Extract voice characteristics from the source video"""
        self.update_progress(30, "Extracting voice audio...")
        
        # Extract audio from video
        audio_path = os.path.join(temp_dir, "source_audio.wav")
        
        try:
            # Use moviepy to extract audio
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            
            # Load audio using librosa for feature extraction
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract voice features
            self.update_progress(33, "Analyzing voice characteristics...")
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pitch (fundamental frequency)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                        fmax=librosa.note_to_hz('C7'), sr=sr)
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Voice activity detection to filter out silence
            intervals = librosa.effects.split(y, top_db=20)
            
            # Combine voiced segments
            voiced_segments = []
            for interval in intervals:
                voiced_segments.append(y[interval[0]:interval[1]])
            
            if voiced_segments:
                # Concatenate voiced segments
                voiced_audio = np.concatenate(voiced_segments)
                
                # Save processed voice
                processed_audio_path = os.path.join(temp_dir, "voice_processed.wav")
                sf.write(processed_audio_path, voiced_audio, sr)
                
                # Package features
                voice_features = {
                    "audio_path": audio_path,
                    "processed_audio_path": processed_audio_path,
                    "mel_spec": mel_spec_db,
                    "mfcc": mfcc,
                    "f0": f0,
                    "sample_rate": sr
                }
                
                self.update_progress(35, "Voice characteristics extracted")
                return voice_features
            else:
                raise ValueError("No voice detected in the video")
                
        except Exception as e:
            self.update_progress(-1, f"Error extracting voice: {str(e)}")
            raise
        
    def generate_synthetic_voice(self, text, voice_features, output_audio_path, language="en", temp_dir=None):
        """Generate synthetic voice saying the provided text"""
        self.update_progress(40, "Generating synthetic voice...")
        
        try:
            # In a real implementation, this would use the voice clone model
            # For now, we'll simulate the process using voice processing techniques
            
            # 1. Text-to-speech (simulated)
            # This would normally use a TTS model to generate initial audio
            self.update_progress(42, "Generating initial speech from text...")
            
            # For simulation, we'll create a simple audio signal
            # In a real implementation, you would use a TTS model
            sr = voice_features["sample_rate"]
            duration = len(text) * 0.1  # rough estimate of duration based on text length
            base_audio = np.sin(2 * np.pi * np.arange(int(sr * duration)) * 220 / sr)
            
            # Save the base audio
            base_audio_path = os.path.join(temp_dir, "base_speech.wav")
            sf.write(base_audio_path, base_audio, sr)
            
            # 2. Voice conversion using the trained model
            self.update_progress(45, "Applying voice characteristics...")
            
            # In a real implementation, this would pass the base audio through the voice clone model
            # For simulation, we'll apply simple processing to make it somewhat unique
            
            # Apply some of the voice characteristics
            # This is a simplified version - real voice cloning would be much more complex
            with torch.no_grad():
                # Process the base audio
                base_audio_tensor = torch.FloatTensor(base_audio).reshape(1, 1, -1).to(self.device)
                
                # Apply the voice clone model
                output_audio = self.voice_clone_model(base_audio_tensor)
                
                # Convert to numpy
                output_audio = output_audio.squeeze().cpu().numpy()
                
                # Normalize
                output_audio = librosa.util.normalize(output_audio)
                
            # Save the result
            sf.write(output_audio_path, output_audio, sr)
            
            self.update_progress(50, "Synthetic voice generated")
            return True
            
        except Exception as e:
            self.update_progress(-1, f"Error generating synthetic voice: {str(e)}")
            return False
        
    def generate_deepfake_video(self, source_video, face_features, text_to_say, 
                               output_audio_path, output_video_path, temp_dir):
        """Generate deepfake video with the person saying the provided text"""
        self.update_progress(55, "Generating deepfake video...")
        
        try:
            # Check if file exists
            if not os.path.exists(source_video):
                raise FileNotFoundError(f"Source video file not found: {source_video}")
                
            # Open the video file
            source_cap = cv2.VideoCapture(source_video)
            if not source_cap.isOpened():
                raise ValueError(f"Could not open source video: {source_video}")
                
            # Load audio to extract timing information
            y, sr = librosa.load(output_audio_path, sr=None)
            
            # Get video properties
            width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = source_cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate how many frames we need based on audio duration
            audio_duration = len(y) / sr
            total_frames = int(audio_duration * fps)
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Process frames to generate the deepfake
            frame_idx = 0
            source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Calculate audio features per frame
            hop_length = int(sr / fps)
            audio_frames = []
            
            for i in range(0, len(y) - hop_length, hop_length):
                if i + hop_length <= len(y):
                    frame_audio = y[i:i + hop_length]
                    audio_frames.append(frame_audio)
            
            # Process frames
            while frame_idx < total_frames and not self.stop_requested:
                # Read frame
                ret, frame = source_cap.read()
                
                # If reached the end of the video, loop back
                if not ret:
                    source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = source_cap.read()
                    if not ret:
                        break
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    # Get the current face location
                    top, right, bottom, left = face_locations[0]
                    
                    # Extract the face region
                    face_image = rgb_frame[top:bottom, left:right]
                    
                    if face_image.size > 0:
                        # Resize for model input
                        face_resized = cv2.resize(face_image, (256, 256))
                        
                        # Get corresponding audio frame
                        audio_frame_idx = min(frame_idx, len(audio_frames) - 1)
                        
                        if audio_frame_idx >= 0 and audio_frame_idx < len(audio_frames):
                            audio_frame = audio_frames[audio_frame_idx]
                            
                            # Prepare inputs for the lip sync model
                            face_tensor = torch.FloatTensor(face_resized).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                            audio_tensor = torch.FloatTensor(audio_frame).unsqueeze(0).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                # Generate lip-synced face
                                lip_synced_face = self.lip_sync_model(face_tensor, audio_tensor)
                                
                                # Convert back to numpy and scale
                                lip_synced_face = lip_synced_face.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                                lip_synced_face = lip_synced_face.astype(np.uint8)
                                
                                # Resize back to original face size
                                lip_synced_face = cv2.resize(lip_synced_face, (right - left, bottom - top))
                                
                                # Blend the lip-synced face with the original frame
                                # For simplicity, here we'll just replace the face region
                                rgb_frame[top:bottom, left:right] = lip_synced_face
                    
                # Convert back to BGR for OpenCV
                output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Add text indication that this is a deepfake
                cv2.putText(output_frame, "DEEPFAKE", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write frame to output
                out.write(output_frame)
                
                # Update progress
                progress = 55 + (frame_idx / total_frames) * 25
                self.update_progress(progress, f"Generating frame {frame_idx+1}/{total_frames}")
                
                frame_idx += 1
            
            # Release resources
            source_cap.release()
            out.release()
            
            # If we stopped early due to user request
            if self.stop_requested:
                return False
                
            self.update_progress(80, "Deepfake video frames generated")
            
            # Combine with audio
            final_output = self.combine_audio_video(output_video_path, output_audio_path, 
                                                   output_video_path.replace('.mp4', '_final.mp4'))
            
            self.update_progress(100, "Deepfake video with audio generated successfully!")
            return final_output
            
        except Exception as e:
            self.update_progress(-1, f"Error generating deepfake video: {str(e)}")
            return False
        
    def combine_audio_video(self, video_path, audio_path, output_path):
        """Combine video and audio using MoviePy"""
        self.update_progress(85, "Combining audio and video...")
        
        try:
            # Load video clip
            video_clip = VideoFileClip(video_path)
            
            # Load audio clip
            audio_clip = AudioFileClip(audio_path)
            
            # Set video clip's audio to the generated audio
            video_clip = video_clip.set_audio(audio_clip)
            
            # Write the result
            video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            # Close clips
            video_clip.close()
            audio_clip.close()
            
            self.update_progress(95, "Audio and video combined")
            return output_path
            
        except Exception as e:
            self.update_progress(-1, f"Error combining audio and video: {str(e)}")
            return False
        
    def process(self, source_video, text_to_say, output_path, language="en"):
        """Main processing pipeline"""
        try:
            # Create temporary directory for intermediate files
            temp_dir = os.path.join(os.path.dirname(output_path), "temp_deepfake")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download and load models
            self.download_models()
            if not self.load_face_model() or not self.load_voice_model():
                return False
            
            # Extract features
            face_features = self.extract_face_features(source_video)
            if self.stop_requested or face_features is None:
                return False
                
            voice_features = self.extract_voice(source_video, temp_dir)
            if self.stop_requested or voice_features is None:
                return False
                
            # Generate audio
            audio_output = os.path.join(temp_dir, "synthetic_voice.wav")
            success = self.generate_synthetic_voice(text_to_say, voice_features, audio_output, language, temp_dir)
            if not success or self.stop_requested:
                return False
                
            # Generate video
            video_output = os.path.join(temp_dir, "deepfake_raw.mp4")
            final_output = self.generate_deepfake_video(source_video, face_features, text_to_say, 
                                                       audio_output, video_output, temp_dir)
            
            if final_output:
                # Make sure the output directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Copy the final output to the desired location
                if os.path.exists(final_output):
                    shutil.copy2(final_output, output_path)
                    final_output = output_path
                    
                # Clean up temp files
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                    
                return final_output
            else:
                return False
                
        except Exception as e:
            self.update_progress(-1, f"Error in processing: {str(e)}")
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
        self.create_widgets
        
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
        ttk.Button(playback_buttons_frame, text="Play Video", 
                  command=self.play_video).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(playback_buttons_frame, text="Open Output Folder", 
                  command=self.open_output_folder).grid(row=0, column=1, padx=5, pady=2)
        
        # Exit button
        ttk.Button(main_frame, text="Exit", command=self.quit).pack(side=tk.RIGHT, pady=10)
        
        # Warning label
        warning_label = ttk.Label(main_frame, text="WARNING: Creating deepfakes may be illegal in some jurisdictions. \nUse responsibly and ethically.", foreground="red")
        warning_label.pack(side=tk.BOTTOM, pady=5)
        
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
            
        # Check required libraries
        try:
            import face_recognition
            import librosa
            import soundfile
            from moviepy.editor import VideoFileClip
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not installed: {str(e)}")
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
            
    def play_video(self):
        """Play the generated video using the default system player"""
        if not self.output_video_path or not os.path.exists(self.output_video_path):
            messagebox.showerror("Error", "No output video available")
            return
            
        try:
            # Use the system's default video player
            if sys.platform == "win32":
                os.startfile(self.output_video_path)
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", self.output_video_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", self.output_video_path])
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not play video: {str(e)}")
            
    def open_output_folder(self):
        """Open the folder containing the output video"""
        if not self.output_video_path or not os.path.exists(self.output_video_path):
            messagebox.showerror("Error", "No output video available")
            return
            
        try:
            # Open the folder containing the file
            output_dir = os.path.dirname(os.path.abspath(self.output_video_path))
            
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", output_dir])
            else:  # Linux
                subprocess.Popen(["xdg-open", output_dir])
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {str(e)}")


def check_requirements():
    """Check if all required libraries are installed"""
    required_libraries = [
        "cv2", 
        "numpy", 
        "torch", 
        "PIL", 
        "face_recognition", 
        "librosa", 
        "soundfile", 
        "moviepy"
    ]
    
    missing_libraries = []
    
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)
            
    if missing_libraries:
        print("Missing required libraries:")
        for lib in missing_libraries:
            print(f"  - {lib}")
            
        print("\nPlease install the missing libraries with:")
        
        # Map library names to pip package names
        pip_packages = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "moviepy": "moviepy",
            "face_recognition": "face-recognition",
            "librosa": "librosa",
            "soundfile": "soundfile"
        }
        
        install_cmd = "pip install " + " ".join([pip_packages.get(lib, lib) for lib in missing_libraries])
        print(install_cmd)
        
        return False
        
    return True


if __name__ == "__main__":
    # Check for required libraries
    if not check_requirements():
        print("Missing required libraries. Please install them before running the application.")
        sys.exit(1)
        
    # Start the application
    app = DeepfakeApp()
    app.mainloop()    