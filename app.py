import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import streamlit as st
import cv2
import tempfile
import ffmpeg
import hashlib
import time
from main import read_video, main, write
from court_detection import CourtDetectionNet
from bounce_detection import BounceDetection
from person_detection import PersonDetection
from ball_detection import BallDetection
from utils import scene_detect
import torch
import sys
from pathlib import Path

# Get the absolute path of the current directory
BASE_DIR = Path(__file__).resolve().parent

# Configuration de la page
st.set_page_config(page_title="Tennis Analysis System", layout="wide")

# Créer un container pour l'état de chargement
loading_container = st.container()

# Initialize models only once at startup
@st.cache_resource
def load_models():
    with loading_container:
        st.markdown("### Initialisation du système d'analyse de tennis")
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if torch.cuda.is_available():
            device = 'cuda'
            st.info("📊 GPU détecté - Utilisation de CUDA pour l'accélération")
        else:
            device = 'cpu'
            st.warning("⚠️ GPU non détecté - Utilisation du CPU (peut être plus lent)")
        
        models = {}
        
        # Créer une barre de progression
        progress_bar = progress_placeholder.progress(0)
        
        try:
            # Utiliser des chemins absolus pour tous les modèles
            # Chargement du modèle de détection de balle
            status_placeholder.markdown("🎾 **Chargement du modèle de détection de balle...**")
            ball_detector_path = BASE_DIR / "Ball detection.pt"
            if not ball_detector_path.exists():
                st.error(f"Fichier modèle introuvable: {ball_detector_path}")
            models['ball_detector'] = BallDetection(str(ball_detector_path), device)
            progress_bar.progress(25)
            
            # Chargement du modèle de détection de court
            status_placeholder.markdown("🎾 **Chargement du modèle de détection de court...**")
            court_detector_path = BASE_DIR / "model_tennis_court_det.pt"
            if not court_detector_path.exists():
                st.error(f"Fichier modèle introuvable: {court_detector_path}")
            models['court_detector'] = CourtDetectionNet(str(court_detector_path), device)
            progress_bar.progress(50)
            
            # Chargement du modèle de détection de personnes
            status_placeholder.markdown("👥 **Chargement du modèle de détection de personnes...**")
            models['person_detector'] = PersonDetection(device)
            progress_bar.progress(75)
            
            # Chargement du modèle de détection de rebonds
            status_placeholder.markdown("💫 **Chargement du modèle de détection de rebonds...**")
            bounce_detector_path = BASE_DIR / "bounce.cbm"
            if not bounce_detector_path.exists():
                st.error(f"Fichier modèle introuvable: {bounce_detector_path}")
            models['bounce_detector'] = BounceDetection(str(bounce_detector_path))
            progress_bar.progress(100)
            
            status_placeholder.markdown("✅ **Tous les modèles sont chargés avec succès !**")
            time.sleep(1)  # Pause pour permettre de voir le message final
        
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles: {str(e)}")
            st.error(f"Chemin d'exécution: {os.getcwd()}")
            st.error(f"Chemin de base: {BASE_DIR}")
            raise e
        finally:
            # Nettoyer les placeholders
            progress_placeholder.empty()
            status_placeholder.empty()
            
        return {
            'device': device,
            **models
        }

# Load models at startup
try:
    models = load_models()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des modèles : {str(e)}")
    st.stop()

@st.cache_data
def get_file_hash(file_bytes):
    """Calculer un hash unique pour le contenu du fichier"""
    return hashlib.sha256(file_bytes).hexdigest()

@st.cache_data
def process_video(_video_bytes, video_hash):
    try:
        # Créer un fichier temporaire avec le contenu de la vidéo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(_video_bytes)
            temp_path = tfile.name

        # Créer un container pour les barres de progression
        progress_container = st.container()
        
        with progress_container:
            # Titre du processus
            st.markdown("### 🎾 Analyse de la vidéo en cours")
            
            # Barre de progression principale
            main_progress = st.progress(0)
            # Message d'état
            status = st.empty()
            # Barre de progression secondaire pour les sous-tâches
            sub_progress = st.progress(0)
            # Message pour les sous-tâches
            sub_status = st.empty()
            
            # Lecture de la vidéo
            status.markdown("📼 **Chargement de la vidéo...**")
            frames, fps = read_video(temp_path)
            main_progress.progress(10)
            status.markdown("✅ Vidéo chargée avec succès")
            
            # Détection des scènes
            status.markdown("🎬 **Analyse des scènes...**")
            sub_status.markdown("Détection des changements de scène")
            scenes = scene_detect(temp_path)
            main_progress.progress(20)
            sub_progress.progress(100)
            status.markdown("✅ Scènes analysées")
            
            # Détection de la balle
            status.markdown("🎾 **Détection de la balle...**")
            sub_status.markdown("Traitement frame par frame")
            ball_track = []
            total_frames = len(frames)
            
            for i, frame in enumerate(frames):
                # Mise à jour de la progression
                sub_progress.progress(int((i + 1) / total_frames * 100))
                sub_status.markdown(f"Frame {i+1}/{total_frames}")
                
            ball_track = models['ball_detector'].infer_model(frames)
            main_progress.progress(40)
            status.markdown("✅ Balle détectée")
            
            # Détection du court
            status.markdown("🏸 **Détection du court...**")
            sub_status.markdown("Analyse de la géométrie du court")
            sub_progress.progress(0)
            homography_matrices, kps_court = models['court_detector'].infer_model(frames)
            main_progress.progress(60)
            sub_progress.progress(100)
            status.markdown("✅ Court détecté")
            
            # Détection des joueurs
            persons_top, persons_bottom = models['person_detector'].track_players(frames, homography_matrices)
            main_progress.progress(80)
            sub_progress.progress(100)
            status.markdown("✅ Joueurs détectés")
            
            # Détection des rebonds
            status.markdown("💫 **Analyse des rebonds...**")
            sub_status.markdown("Calcul des trajectoires")
            x_ball = [x[0] for x in ball_track]
            y_ball = [x[1] for x in ball_track]
            bounces = models['bounce_detector'].predict(x_ball, y_ball)
            main_progress.progress(90)
            status.markdown("✅ Rebonds analysés")
            
            # Génération de la vidéo finale
            status.markdown("🎥 **Génération de la vidéo finale...**")
            sub_status.markdown("Application des annotations")
            imgs_res = main(frames, scenes, bounces, ball_track, homography_matrices, 
                          kps_court, persons_top, persons_bottom, draw_trace=True)
            
            # Ensure output directory exists
            output_dir = BASE_DIR / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Sauvegarder en MP4
            output_path = output_dir / f"output_{video_hash[:8]}.mp4"
            if output_path.exists():
                output_path.unlink()
                
            # Créer un fichier temporaire pour les images
            temp_dir = output_dir / f"temp_frames_{video_hash[:8]}"
            temp_dir.mkdir(exist_ok=True)
            
            # Sauvegarder les images
            sub_status.markdown("Encodage de la vidéo")
            for i, frame in enumerate(imgs_res):
                cv2.imwrite(str(temp_dir / f"frame_{i:04d}.png"), frame)
                sub_progress.progress(int((i + 1) / len(imgs_res) * 100))
            
            try:
                # Check if ffmpeg is available
                import shutil
                ffmpeg_path = shutil.which("ffmpeg")
                
                # Define common FFmpeg install locations to check
                potential_ffmpeg_paths = [
                    BASE_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",  # Local to project
                    BASE_DIR / "ffmpeg.exe",                     # Direct in project root
                    Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')) / "ffmpeg" / "bin" / "ffmpeg.exe",
                    Path(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')) / "ffmpeg" / "bin" / "ffmpeg.exe",
                ]
                
                # Check all potential paths
                if not ffmpeg_path:
                    for path in potential_ffmpeg_paths:
                        if path.exists():
                            ffmpeg_path = str(path)
                            st.info(f"FFmpeg found at alternative location: {ffmpeg_path}")
                            os.environ["PATH"] += os.pathsep + str(path.parent)
                            break
                
                # Use FFmpeg if available
                if ffmpeg_path:
                    st.info(f"FFmpeg found at: {ffmpeg_path}")
                    
                    # Utiliser FFmpeg pour créer la vidéo
                    input_pattern = str(temp_dir / "frame_%04d.png")
                    output_path_str = str(output_path)
                    st.info(f"Input pattern: {input_pattern}")
                    st.info(f"Output path: {output_path_str}")
                    
                    stream = ffmpeg.input(input_pattern, framerate=fps)
                    stream = ffmpeg.output(stream, output_path_str, 
                                         vcodec='libx264',
                                         preset='medium',
                                         crf='23',
                                         movflags='faststart',
                                         pix_fmt='yuv420p')
                    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                else:
                    # Define output_path_str for all code paths
                    output_path_str = str(output_path)
                    
                    # Try to use FFmpeg from the specific path the user has it installed
                    ffmpeg_exe_path = Path("C:/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe")
                    
                    # Add the bin directory to the PATH so ffmpeg-python can find it
                    if ffmpeg_exe_path.exists():
                        st.info(f"Found FFmpeg at: {ffmpeg_exe_path}")
                        # Add to environment PATH
                        ffmpeg_bin_dir = str(ffmpeg_exe_path.parent)
                        os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ["PATH"]
                        
                        try:
                            # Try using ffmpeg-python again now that we've updated the PATH
                            input_pattern = str(temp_dir / "frame_%04d.png")
                            stream = ffmpeg.input(input_pattern, framerate=fps)
                            stream = ffmpeg.output(stream, output_path_str, 
                                                vcodec='libx264',
                                                preset='medium',
                                                crf='23',
                                                movflags='faststart',
                                                pix_fmt='yuv420p')
                            st.info("Trying ffmpeg-python again with updated PATH")
                            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                            st.success("Video created successfully with FFmpeg")
                        except Exception as e:
                            st.error(f"Error with ffmpeg-python after PATH update: {str(e)}")
                            
                            # Fall back to direct subprocess call
                            try:
                                import subprocess
                                ffmpeg_cmd = [
                                    str(ffmpeg_exe_path),
                                    "-y",  # Overwrite output files
                                    "-framerate", str(fps),
                                    "-i", str(temp_dir / "frame_%04d.png"),
                                    "-c:v", "libx264",
                                    "-preset", "medium",
                                    "-crf", "23",
                                    "-pix_fmt", "yuv420p",
                                    output_path_str
                                ]
                                st.info(f"Running direct FFmpeg command: {' '.join(ffmpeg_cmd)}")
                                result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                                st.success("Video created successfully with direct FFmpeg call")
                            except Exception as e:
                                st.error(f"Error with direct FFmpeg call: {str(e)}")
                                # Fall back to OpenCV as last resort
                                use_opencv = True
                    else:
                        st.error(f"FFmpeg not found at the expected location: {ffmpeg_exe_path}")
                        use_opencv = True
                    
                    # Fallback to OpenCV VideoWriter if all FFmpeg attempts fail
                    if 'use_opencv' in locals() and use_opencv:
                        st.warning("Falling back to OpenCV VideoWriter...")
                        
                        height, width = imgs_res[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
                        out = cv2.VideoWriter(output_path_str, fourcc, fps, (width, height))
                        
                        # Write frames to video
                        for i, frame in enumerate(imgs_res):
                            out.write(frame)
                            sub_progress.progress(int((i + 1) / len(imgs_res) * 100))
                        
                        out.release()
                        st.success("Video created successfully with OpenCV")
                
                # Nettoyer les fichiers temporaires
                import shutil
                shutil.rmtree(str(temp_dir))
                
                main_progress.progress(100)
                status.markdown("✅ **Traitement terminé avec succès !**")
                sub_status.empty()
                sub_progress.empty()
                
                # Lire le fichier de sortie
                with open(output_path_str, 'rb') as f:
                    video_bytes = f.read()
                
                return video_bytes
                
            except ffmpeg.Error as e:
                st.error(f"Erreur FFmpeg : {str(e)}")
                if temp_dir.exists():
                    shutil.rmtree(str(temp_dir))
                return None
            
    except Exception as e:
        import traceback
        st.error(f"Erreur lors du traitement : {str(e)}")
        st.error(traceback.format_exc())
        return None
    finally:
        # Nettoyer le fichier temporaire d'entrée
        if os.path.exists(temp_path):
            os.unlink(temp_path)

st.title("Tennis Analysis System")
st.write("Upload a tennis video to detect ball, court, players and bounces")

# File uploader
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

if video_file is not None:
    # Lire le contenu du fichier une seule fois
    video_bytes = video_file.read()
    # Calculer un hash unique pour le fichier
    video_hash = get_file_hash(video_bytes)
    
    # Traiter la vidéo avec le hash comme clé de cache
    processed_video = process_video(video_bytes, video_hash)
    
    if processed_video:
        # Afficher la vidéo
        st.video(processed_video)
        
        # Ajouter le bouton de téléchargement
        st.download_button(
            label="Download processed video",
            data=processed_video,
            file_name="tennis_analysis.mp4",
            mime="video/mp4"
        )