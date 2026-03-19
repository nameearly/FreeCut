import sys
import os
import json
import numpy as np
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, vfx, afx
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

def get_speed_interpolator(speed_kfs):
    if not speed_kfs: return lambda t: t
    kfs = sorted(speed_kfs, key=lambda x: x['time'])
    times = [kf['time'] for kf in kfs]
    values = [kf['value'] for kf in kfs]

    def timeline_to_asset_time(t):
        # t pode ser um número ou um array de números (comum no áudio do MoviePy)
        is_array = isinstance(t, np.ndarray)
        t_func = t if is_array else np.array([t])
        
        results = []
        for val in t_func:
            if val <= times[0]:
                results.append(val * values[0])
                continue
            
            acc = times[0] * values[0]
            found = False
            for i in range(len(times) - 1):
                t_s, t_e = times[i], times[i+1]
                v_s, v_e = values[i], values[i+1]
                if val > t_e:
                    acc += (t_e - t_s) * (v_s + v_e) / 2
                else:
                    dt = val - t_s
                    v_curr = np.interp(val, [t_s, t_e], [v_s, v_e])
                    acc += dt * (v_s + v_curr) / 2
                    results.append(acc)
                    found = True
                    break
            if not found:
                results.append(acc + (val - times[-1]) * values[-1])
        
        return np.array(results) if is_array else results[0]

    return timeline_to_asset_time

def export_video():
    try:
        if len(sys.argv) < 2: sys.exit(1)
        config_path = sys.argv[1]
        with open(config_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            
        export_path = payload['export_path']
        clips_data = payload['clips']
        target_size = (1920, 1080) 

        video_clips = []
        for c in clips_data:
            path = c['path']
            
            # 1. Carregamento e Speed Ramp
            if c.get('type') == 'image':
                clip = ImageClip(path).with_duration(c['duration'])
            else:
                full_clip = VideoFileClip(path)
                speed_kfs = c.get('keyframes', {}).get('speed', [])
                if speed_kfs:
                    time_mapper = get_speed_interpolator(speed_kfs)
                    clip = full_clip.time_transform(time_mapper)
                    if full_clip.audio is not None:
                        clip = clip.with_audio(full_clip.audio.time_transform(time_mapper))
                    clip = clip.with_duration(c['duration'])
                    clip = clip.subclipped(c['beginmoment'], c['beginmoment'] + c['duration'])
                else:
                    clip = full_clip.subclipped(c['beginmoment'], c['beginmoment'] + c['duration'])

            # 2. Fades
            f_in = float(c.get("fadein", 0))
            f_out = float(c.get("fadeout", 0))
            fa_in = float(c.get("fadeinAudio", 0))
            fa_out = float(c.get("fadeoutAudio", 0))

            if f_in > 0: clip = clip.with_effects([vfx.FadeIn(f_in)])
            if f_out > 0: clip = clip.with_effects([vfx.FadeOut(f_out)])
            if clip.audio is not None:
                if fa_in > 0: clip.audio = clip.audio.with_effects([afx.AudioFadeIn(fa_in)])
                if fa_out > 0: clip.audio = clip.audio.with_effects([afx.AudioFadeOut(fa_out)])

            # 3. Keyframes de Opacidade
            op_kfs = c.get('keyframes', {}).get('opacity', [])
            if op_kfs:
                op_kfs = sorted(op_kfs, key=lambda x: x['time'])
                o_times = [kf['time'] for kf in op_kfs]
                o_values = [kf['value'] for kf in op_kfs]
                
                def apply_opacity(get_f, t):
                    frame = get_f(t)
                    # np.interp lida bem com t sendo array ou escalar
                    val = np.interp(t, o_times, o_values)
                    # Se for áudio/array, precisamos expandir dimensões para multiplicar pelo frame
                    if isinstance(val, np.ndarray):
                        return (frame * val[:, None, None, None]).astype('uint8')
                    return (frame * val).astype('uint8')
                clip = clip.transform(apply_opacity)

            # 4. Keyframes de Volume
            vol_kfs = c.get('keyframes', {}).get('volume', [])
            if (c.get('type') != 'image') and vol_kfs and clip.audio is not None:
                vol_kfs = sorted(vol_kfs, key=lambda x: x['time'])
                v_times = [kf['time'] for kf in vol_kfs]
                v_values = [kf['value'] for kf in vol_kfs]
                
                def apply_volume(get_f, t):
                    chunk = get_f(t)
                    val = np.interp(t, v_times, v_values)
                    gain = 10 ** ((-30 + (val * 60)) / 20)
                    
                    if isinstance(gain, np.ndarray):
                        return chunk * gain[:, np.newaxis]
                    return chunk * gain
                clip.audio = clip.audio.transform(apply_volume)

            # 5. Redimensionamento
            clip = clip.resized(height=target_size[1])
            if clip.w > target_size[0]: clip = clip.resized(width=target_size[0])
            clip = clip.with_start(c['start']).with_position("center")
            video_clips.append(clip)

        # 6. Escrita do arquivo
        final_video = CompositeVideoClip(video_clips, size=target_size)
        final_video.write_videofile(
            export_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            logger=RawPercentageLogger()
        )
        sys.stderr.write("PERCENT:100\n")
        sys.exit(0)

    except Exception as e:
        sys.stderr.write(f"ERRO: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    export_video()