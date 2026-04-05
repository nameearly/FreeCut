import json
import sys
import os
import numpy as np
import cv2
import shutil
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy import VideoFileClip, AudioFileClip, ImageClip, CompositeAudioClip
from moviepy.video.VideoClip import VideoClip
from proglog import ProgressBarLogger
from PIL import Image, ImageDraw, ImageFont

# --- LOGGER ---
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

def generate_text_frame(clip_data):
    # Pega o texto do campo 'name' (ou 'text' se você preferir mudar no JSON)
    text = clip_data.get('name', 'Texto')
    
    bw = int(clip_data.get('bg_dimetions', {}).get('x', 400))
    bh = int(clip_data.get('bg_dimetions', {}).get('y', 100))
    
    # Resolução interna alta para nitidez (4x)
    canvas = Image.new('RGBA', (bw * 4, bh * 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    bg_color = clip_data.get('font_bgcolor', 'transparent')
    if bg_color != 'transparent':
        draw.rectangle([0, 0, canvas.width, canvas.height], fill=bg_color)
    
    f_size = int((clip_data.get('font_size') or 20) * 4)
    
    # --- CORREÇÃO AQUI ---
    font_path = clip_data.get('font') # Agora é o caminho completo enviado pelo TS
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, f_size)
        else:
            font = ImageFont.truetype("Arial", f_size) # Fallback
    except:
        font = ImageFont.load_default()
    # ---------------------
    
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((canvas.width-(right-left))/2, (canvas.height-(bottom-top))/2 - top), 
              text, font=font, fill=clip_data.get('font_color', '#ffffff'))
    
    return np.array(canvas)
def process_video():
    if len(sys.argv) < 2: return
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)

    project_path = data['project_path']
    PROJ_W, PROJ_H = data['project_dimensions']['width'], data['project_dimensions']['height']
    clips_data = data['clips'][::-1]
    loaded_clips = []
    
    for c in clips_data:
        if c['type'] == 'text':
            img = generate_text_frame(c)
            loaded_clips.append({'data': c, 'text_image': img})
            continue
        
        path = os.path.join(project_path, "videos", c['name'])
        ext = os.path.splitext(c['name'])[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp']:
            v = ImageClip(path).with_duration(c['duration'])
            loaded_clips.append({'data': c, 'video': v})
        elif c['type'] == 'video':
            v = VideoFileClip(path, audio=not c.get('mute'))
            v = v.subclipped(c.get('beginmoment', 0), c.get('beginmoment', 0) + c['duration'])
            loaded_clips.append({'data': c, 'video': v})
        elif c['type'] == 'audio':
            a = AudioFileClip(path).subclipped(c.get('beginmoment', 0), c.get('beginmoment', 0) + c['duration'])
            loaded_clips.append({'data': c, 'audio': a})

    # ... (mantenha os imports e classes iniciais iguais)

    def make_final_frame(t):
        canvas_f = np.zeros((PROJ_H, PROJ_W, 3), dtype=float)
        for item in loaded_clips:
            c = item['data']
            rel_t = t - c['start']
            if rel_t < 0 or rel_t >= c['duration']: continue

            if 'text_image' in item:
                img_rgba = item['text_image'].copy()
                cw, ch = c['bg_dimetions']['x'], c['bg_dimetions']['y']
                # Para texto, a escala base é 1.0 para manter o tamanho do canvas gerado
                is_text = True
            elif 'video' in item:
                img_rgba = cv2.cvtColor(item['video'].get_frame(rel_t), cv2.COLOR_RGB2RGBA)
                cw, ch = item['video'].w, item['video'].h
                is_text = False
            else: continue

            # --- CORREÇÃO DA LÓGICA DE ESCALA ---
            if is_text:
                base_scale = 1.0
            else:
                scale_x = PROJ_W / cw
                scale_y = PROJ_H / ch
                base_scale = max(scale_x, scale_y) 

            kfs = c.get('keyframes', {})
            op = float(get_interpolated_value(kfs.get('opacity', []), rel_t, 1.0))
            zoom = float(get_interpolated_value(kfs.get('zoom', []), rel_t, 1.0))
            rot = get_interpolated_value(kfs.get('rotation3d', []), rel_t, {"rot": 0, "rot3d": 0})
            pos = get_interpolated_value(kfs.get('position', []), rel_t, {"x": 0, "y": 0})

            # Redimensionamento
            fw, fh = int(cw * base_scale * zoom), int(ch * base_scale * zoom)
            img_resized = cv2.resize(img_rgba, (fw, fh), interpolation=cv2.INTER_LINEAR)
            img_final = apply_3d_rotation(img_resized, rot['rot'], -rot['rot3d'])
            
            # --- CORREÇÃO DO POSICIONAMENTO (Sincronizado com o React) ---
            h_f, w_f = img_final.shape[:2]
            
            if is_text:
                # Replicando os offsets: pos.x - 20 e pos.y - 120 + (PROJ_H / 2)
                # Como no Python o 0,0 é topo-esquerda e no Three.js é centralizado,
                # ajustamos para bater com a visualização do canvas
                x1 = int(pos['x'] - 20)
                y1 = int(pos['y'] - 120 + (PROJ_H / 2))
            else:
                x1 = int(pos['x'])
                y1 = int(pos['y'])

            # Composição com o Canvas
            ix1, ix2 = max(0, x1), min(PROJ_W, x1 + w_f)
            iy1, iy2 = max(0, y1), min(PROJ_H, y1 + h_f)
            
            if ix1 < ix2 and iy1 < iy2:
                fx1, fy1 = ix1 - x1, iy1 - y1
                src_crop = img_final[fy1:fy1+(iy2-iy1), fx1:fx1+(ix2-ix1)].astype(float) / 255.0
                
                # Multiplica o canal Alpha da imagem pela opacidade do clipe
                alpha = src_crop[:, :, 3:4] * op
                
                tgt = canvas_f[iy1:iy2, ix1:ix2]
                # Blend manual: (NovaCor * Alpha) + (CorFundo * (1 - Alpha))
                canvas_f[iy1:iy2, ix1:ix2] = (src_crop[:,:,:3] * alpha) + (tgt * (1.0 - alpha))

        return (canvas_f * 255).astype('uint8')


        
    duration = max((c['start'] + c['duration']) for c in clips_data) if clips_data else 0
    final_video = FreeCutVideoClip(make_final_frame, duration, (PROJ_W, PROJ_H))
    
    tracks = []
    for item in loaded_clips:
        if 'audio' in item: tracks.append(item['audio'].with_start(item['data']['start']))
        elif 'video' in item and hasattr(item['video'], 'audio') and item['video'].audio and not item['data'].get('mute'):
            tracks.append(item['video'].audio.with_start(item['data']['start']))
    if tracks: final_video.audio = CompositeAudioClip(tracks)

    final_video.write_videofile(data['export_path'], fps=30, codec="libx264", audio_codec="aac", logger=RawPercentageLogger())

if __name__ == "__main__":
    process_video()