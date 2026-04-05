import json
import sys
import os
import numpy as np
import cv2
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy import VideoFileClip, AudioFileClip, ImageClip, CompositeAudioClip
from moviepy.video.VideoClip import VideoClip
from proglog import ProgressBarLogger
from PIL import Image, ImageDraw, ImageFont

class RawPercentageLogger(ProgressBarLogger):
    def __init__(self):
        super().__init__()
        self.last_percentage = -1

    def callback(self, **changes):
        bars = self.state.get('bars', {})
        if not bars: return
        bar_list = list(bars.values())
        if not bar_list: return
        current_bar = bar_list[-1]
        if current_bar.get('total', 0) > 0 and current_bar.get('title') != 'chunk':
            percent = int((current_bar.get('index', 0) / current_bar.get('total')) * 100)
            if percent != self.last_percentage:
                sys.stderr.write(f"PERCENT:{min(100, percent)}\n")
                sys.stderr.flush()
                self.last_percentage = percent

class FreeCutVideoClip(VideoClip):
    def __init__(self, make_frame, duration, size):
        super().__init__()
        self.make_frame = make_frame
        self.frame_function = make_frame
        self.duration = duration
        self.end = duration
        self.size = size

def get_interpolated_value(keyframes, t, default_value):
    if not keyframes or not isinstance(keyframes, list) or len(keyframes) == 0:
        return default_value
    sorted_kfs = sorted(keyframes, key=lambda x: x['time'])
    kf_times = [float(kf['time']) for kf in sorted_kfs]
    if t <= kf_times[0]: return sorted_kfs[0]['value']
    if t >= kf_times[-1]: return sorted_kfs[-1]['value']
    for i in range(len(kf_times) - 1):
        if kf_times[i] <= t <= kf_times[i+1]:
            t1, t2 = kf_times[i], kf_times[i+1]
            v1, v2 = sorted_kfs[i]['value'], sorted_kfs[i+1]['value']
            f = (t - t1) / (t2 - t1)
            if isinstance(v1, dict):
                res = {}
                for k in v1.keys():
                    res[k] = float(v1[k]) + (float(v2[k]) - float(v1[k])) * f
                return res
            return float(v1) + (float(v2) - float(v1)) * f
    return default_value

def apply_3d_rotation(img_rgba, rot_deg, rot3d_deg):
    h, w = img_rgba.shape[:2]
    center = (w / 2, h / 2)
    matrix_2d = cv2.getRotationMatrix2D(center, -float(rot_deg), 1.0)
    img_rgba = cv2.warpAffine(img_rgba, matrix_2d, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    if abs(float(rot3d_deg)) < 0.1: return img_rgba
    rad = np.radians(float(rot3d_deg))
    fov = 0.3
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dist_w = (w / 2) * np.cos(rad)
    off = (w / 2) * np.sin(rad) * fov
    dst_pts = np.float32([[center[0]-dist_w, 0+off], [center[0]+dist_w, 0-off], [center[0]+dist_w, h+off], [center[0]-dist_w, h-off]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def generate_text_frame(clip_data, proj_w, proj_h):
    # Se o clipe tem bg_dimetions minúsculo ou nulo, usamos o padrão do projeto para manter a escala
    user_w = clip_data.get('bg_dimetions', {}).get('x') if clip_data.get('bg_dimetions') else None
    user_h = clip_data.get('bg_dimetions', {}).get('y') if clip_data.get('bg_dimetions') else None
    
    # Lógica de fallback: se for menor que 512, tratamos como um canvas de texto padrão para não achatar
    bw = int(user_w) if user_w and user_w > 511 else proj_w
    bh = int(user_h) if user_h and user_h > 100 else int(proj_h / 4)
    
    # Criamos o canvas em alta resolução (x2) para garantir nitidez extrema
    render_w, render_h = bw * 2, bh * 2
    pil_img = Image.new('RGBA', (render_w, render_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    
    bg_color = clip_data.get('font_bgcolor', 'transparent')
    if bg_color and bg_color != 'transparent':
        draw.rectangle([0, 0, render_w, render_h], fill=bg_color)
    
    # Multiplicador de fonte: ajustado para o supersampling (x2)
    f_size = int(clip_data.get('font_size') or 14) * 8
    font_path = clip_data.get('font')
    
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, f_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    text = clip_data.get('name', 'Text')
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    
    draw.text(
        ((render_w - tw) / 2, (render_h - th) / 2 - top),
        text, font=font, fill=clip_data.get('font_color', '#ffffff')
    )
    
    # Reduzimos para o tamanho de destino (bw, bh) para antialiasing natural
    cv2_img = np.array(pil_img)
    cv2_img = cv2.resize(cv2_img, (bw, bh), interpolation=cv2.INTER_AREA)
    return cv2_img

def process_video():
    if len(sys.argv) < 2: return
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)

    project_path = data['project_path']
    PROJ_W, PROJ_H = data['project_dimensions']['width'], data['project_dimensions']['height']
    # Mantemos a ordem inversa para que o Track 0/1 (topo) seja desenhado por último
    clips_data = sorted(data['clips'], key=lambda x: x.get('trackId', 0), reverse=True)
    loaded_clips = []
    
    for c in clips_data:
        if c['type'] == 'text':
            img = generate_text_frame(c, PROJ_W, PROJ_H)
            loaded_clips.append({'data': c, 'type': 'text', 'image': img})
        else:
            path = os.path.join(project_path, "videos", c['name'])
            if not os.path.exists(path): continue
            ext = os.path.splitext(c['name'])[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.webp']:
                v = ImageClip(path).with_duration(c['duration'])
                loaded_clips.append({'data': c, 'type': 'media', 'video': v})
            elif c['type'] == 'video':
                v = VideoFileClip(path, audio=not c.get('mute'))
                v = v.subclipped(c.get('beginmoment', 0), c.get('beginmoment', 0) + c['duration'])
                loaded_clips.append({'data': c, 'type': 'media', 'video': v})

    def make_final_frame(t):
        canvas_f = np.zeros((PROJ_H, PROJ_W, 3), dtype=float)
        
        for item in loaded_clips:
            c = item['data']
            rel_t = t - c['start']
            if rel_t < 0 or rel_t >= c['duration']: continue

            if item['type'] == 'text':
                img_rgba = item['image'].copy()
                is_text = True
            else:
                img_rgba = cv2.cvtColor(item['video'].get_frame(rel_t), cv2.COLOR_RGB2RGBA)
                is_text = False

            cw, ch = img_rgba.shape[1], img_rgba.shape[0]
            kfs = c.get('keyframes', {})
            op = float(get_interpolated_value(kfs.get('opacity', []), rel_t, 1.0))
            zoom = float(get_interpolated_value(kfs.get('zoom', []), rel_t, 1.0))
            rot = get_interpolated_value(kfs.get('rotation3d', []), rel_t, {"rot": 0, "rot3d": 0})
            pos = get_interpolated_value(kfs.get('position', []), rel_t, {"x": 0, "y": 0})

            if is_text:
                fw, fh = int(cw * zoom), int(ch * zoom)
            else:
                scale_fit = max(PROJ_W / cw, PROJ_H / ch)
                fw, fh = int(cw * scale_fit * zoom), int(ch * scale_fit * zoom)

            img_resized = cv2.resize(img_rgba, (max(1, fw), max(1, fh)), interpolation=cv2.INTER_LINEAR)
            img_transformed = apply_3d_rotation(img_resized, rot['rot'], -rot['rot3d'])
            tw, th = img_transformed.shape[1], img_transformed.shape[0]
            
            # ORIGEM NO CENTRO (Igual ao Three.js)
            # PROJ_W/2 e PROJ_H/2 definem o centro. pos['x'] e pos['y'] deslocam a partir dali.
            # tw/2 e th/2 garantem que a imagem seja desenhada a partir do seu próprio centro.
            x1 = int((PROJ_W / 2) + pos['x'] - (tw / 2))
            y1 = int((PROJ_H / 2) + pos['y'] - (th / 2))

            ix1, ix2 = max(0, x1), min(PROJ_W, x1 + tw)
            iy1, iy2 = max(0, y1), min(PROJ_H, y1 + th)
            
            if ix1 < ix2 and iy1 < iy2:
                fx1, fy1 = ix1 - x1, iy1 - y1
                src_crop = img_transformed[fy1:fy1+(iy2-iy1), fx1:fx1+(ix2-ix1)].astype(float) / 255.0
                alpha = src_crop[:, :, 3:4] * op
                
                tgt = canvas_f[iy1:iy2, ix1:ix2]
                if c.get('blendmode') == 'screen':
                    canvas_f[iy1:iy2, ix1:ix2] = 1.0 - (1.0 - src_crop[:,:,:3] * alpha) * (1.0 - tgt)
                else:
                    canvas_f[iy1:iy2, ix1:ix2] = (src_crop[:,:,:3] * alpha) + (tgt * (1.0 - alpha))

        return (canvas_f * 255).astype('uint8')

    duration = max((c['start'] + c['duration']) for c in clips_data) if clips_data else 0
    final_video = FreeCutVideoClip(make_final_frame, duration, (PROJ_W, PROJ_H))
    
    tracks = []
    for item in loaded_clips:
        if item['type'] == 'media' and hasattr(item['video'], 'audio') and item['video'].audio and not item['data'].get('mute'):
            tracks.append(item['video'].audio.with_start(item['data']['start']))
    if tracks: final_video.audio = CompositeAudioClip(tracks)

    final_video.write_videofile(data['export_path'], fps=30, codec="libx264", audio_codec="aac", logger=RawPercentageLogger())

if __name__ == "__main__":
    process_video()