from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_stream import VideoOpenFailure
from scenedetect import open_video
from post_process import PostProcess

def scene_detect(path_video, threshold=30.0):
    """
    Split video to disjoint fragments based on color histograms using ContentDetector.
    :param path_video: Path to the video file.
    :param threshold: Threshold for ContentDetector (default: 30.0).
    :return: List of scenes, where each scene is [start_frame, end_frame].
    """
    scenes = []
    video = None
    try:
        video = open_video(path_video)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        base_timecode = video.base_timecode

        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            end_time = video.duration
            if end_time:
                scene_list = [(base_timecode, end_time)]
            else:
                print("Warning: Could not get video duration, returning empty scene list.")
                return []

        scenes = [[x[0].frame_num, x[1].frame_num] for x in scene_list]

        total_frames = video.frame_number
        if scenes and total_frames > 0 and scenes[-1][1] < total_frames:
            scenes[-1][1] = min(scenes[-1][1], total_frames)

    except VideoOpenFailure as e:
        print(f"Error opening video: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during scene detection: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        if video is not None:
            pass

    return scenes