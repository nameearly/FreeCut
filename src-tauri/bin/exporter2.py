import json
import sys
import os
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, vfx, afx
import numpy as np

from moviepy.video.VideoClip import VideoClip
from proglog import ProgressBarLogger

class RawPercentageLogger(ProgressBarLogger):
    def __init__(self):
        super().__init__()
        self.last_percentage = -1
        self.rendering_video_finished = False
        self.startzero = False

    def callback(self, **changes):
        bars = self.state.get('bars', {})
        if not bars: return
        current_bar = list(bars.values())[-1]
        total = current_bar.get('total', 0)
        index = current_bar.get('index', 0)
        if total > 0:
            percent = int((index / total) * 100)
            if percent >= 100: percent = 99
            if percent != self.last_percentage:
                if not self.rendering_video_finished:
                    if percent == 99: self.rendering_video_finished = True
                    return 
                if not self.startzero:
                    if percent == 0: self.startzero = True
                    return
                sys.stderr.write(f"PERCENT:{percent}\n")
                sys.stderr.flush()
                self.last_percentage = percent


def get_interpolated_value_array(keyframes, current_times, default_value):
    if not keyframes:
        return default_value
    kf_times = np.array([kf['time'] for kf in keyframes])
    kf_values = np.array([kf['value'] for kf in keyframes])
    indices = np.argsort(kf_times)
    return np.interp(current_times, kf_times[indices], kf_values[indices], left=kf_values[indices][0], right=kf_values[indices][-1])

def get_pos_interpolated(keyframes, t, default=(0,0)):
    if not keyframes: return default
    kf_times = np.array([kf['time'] for kf in keyframes])
    if len(kf_times) == 0: return default
    indices = np.argsort(kf_times)
    kf_times = kf_times[indices]
    kf_x = np.array([kf['value']['x'] for kf in keyframes])[indices]
    kf_y = np.array([kf['value']['y'] for kf in keyframes])[indices]
    return (float(np.interp(t, kf_times, kf_x)), float(np.interp(t, kf_times, kf_y)))

def process_video():
    if len(sys.argv) < 2:
        print("Uso: python exporter2.py <caminho_do_json>")
        return

    json_path = sys.argv[1]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    project_w = data['project_dimensions']['width']
    project_h = data['project_dimensions']['height']
    clips_data = data['clips']
    
    final_clips = []

    for c_data in clips_data:
        path = c_data['path']
        start_time = c_data['start']
        duration = c_data['duration']
        begin_moment = c_data.get('beginmoment', 0)
        current_kf = c_data.get('keyframes', {})
        f_in = c_data.get('fadein', 0)
        f_out = c_data.get('fadeout', 0)

        if c_data['type'] == 'video':
            clip = VideoFileClip(path).subclipped(begin_moment, begin_moment + duration)
            
            if c_data.get('scale', 1) != 1:
                clip = clip.resized(c_data['scale'])
            
            if c_data.get('mute', False):
                clip = clip.without_audio()

            # --- MÁSCARA DINÂMICA (API 2.0 SEGURA) ---
            h, w = clip.h, clip.w

            def make_mask_frame(t, kf=current_kf, fin=f_in, fout=f_out, dur=duration, ch=h, cw=w):
                op = get_interpolated_value_array(kf.get('opacity', []), t, 1.0)
                
                if isinstance(t, np.ndarray):
                    fade_mult = np.ones_like(t)
                    if fin > 0: fade_mult *= np.minimum(1.0, t / fin)
                    if fout > 0: fade_mult *= np.minimum(1.0, (dur - t) / fout)
                    op *= fade_mult
                    mask = np.zeros((len(t), ch, cw))
                    for i in range(len(t)):
                        mask[i] = op[i]
                    return mask
                else:
                    if fin > 0 and t < fin: op *= (t / fin)
                    if fout > 0 and t > (dur - fout): op *= ((dur - t) / fout)
                    return np.full((ch, cw), op, dtype=float)

            # Criando o clip de máscara usando o método de inicialização mais robusto da 2.0
            my_mask = VideoClip(is_mask=True).with_updated_frame_function(make_mask_frame).with_duration(clip.duration)
            clip = clip.with_mask(my_mask)

            # --- POSIÇÃO ---
            def make_pos(t, kf=current_kf):
                return get_pos_interpolated(kf.get('position', []), t)
            
            clip = clip.with_position(make_pos)

        else: # Audio
            clip = AudioFileClip(path).subclipped(begin_moment, begin_moment + duration)

        # --- ÁUDIO ---
        if clip.audio:
            vol_kf = current_kf.get('volume', [])
            fa_in = c_data.get('fadeinAudio', 0)
            fa_out = c_data.get('fadeoutAudio', 0)
            
            def vol_transform(get_f, t, vkf=vol_kf, fin=fa_in, fout=fa_out, dur=duration):
                res = get_f(t)
                volumes = get_interpolated_value_array(vkf, t, 1.0)
                if isinstance(t, np.ndarray):
                    fade_mult = np.ones_like(t)
                    if fin > 0: fade_mult *= np.minimum(1.0, t / fin)
                    if fout > 0: fade_mult *= np.minimum(1.0, (dur - t) / fout)
                    volumes *= fade_mult
                    return res * volumes[:, np.newaxis]
                else:
                    if fin > 0 and t < fin: volumes *= (t / fin)
                    if fout > 0 and t > (dur - fout): volumes *= ((dur - t) / fout)
                    return res * volumes

            clip.audio = clip.audio.transform(vol_transform)

        clip = clip.with_start(start_time)
        final_clips.append(clip)

    video_final = CompositeVideoClip(final_clips, size=(project_w, project_h))
    
    output_path = data['export_path']
    temp_dir = data['project_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_final.write_videofile(
        output_path, 
        fps=30, 
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile=os.path.join(temp_dir, "temp-audio.m4a"),
        remove_temp=True,
        logger=RawPercentageLogger()
    )

if __name__ == "__main__":
    process_video()