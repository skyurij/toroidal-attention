import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from PIL import Image, ImageDraw, ImageFont
import wave

# ============================================
# CONFIGURATION (READER: SONNET)
# ============================================

AUDIO_PATH = "sonnet_1.wav"
TRANSCRIPT_PATH = "sonnet_1.txt"

FPS = 30

R = 1.4          # большой радиус тора
r = 0.55         # малый радиус тора
FOLLOWER_SMOOTH = 0.12

LINE_WIDTH = 1.5
FONT_SIZE = 18

COLORS = [
    "#66ccff",
    "#cc66ff",
    "#55ffcc",
    "#ffaa55",
    "#ff6699",
]

CYCLE_2PI = 2 * np.pi

SAVE_FRAMES = True
FRAMES_DIR = "frames_reader"
os.makedirs(FRAMES_DIR, exist_ok=True)

TEXT_CYCLES = 4.0
AUDIO_CYCLES = 4.0

MAIN_AXIS_GAIN = 0.02


# ============================================
# PHASE FROM AUDIO AND TEXT
# ============================================

def load_audio_phase(audio_path, fps, cycles=AUDIO_CYCLES):
    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    audio = audio - np.mean(audio)
    max_abs = np.max(np.abs(audio)) + 1e-9
    audio = audio / max_abs

    env = np.abs(audio)

    win = max(500, len(env) // 200)
    if win > 1:
        kernel = np.ones(win, dtype=np.float32) / win
        env = np.convolve(env, kernel, mode="same")

    env = env + 1e-6

    duration_sec = len(audio) / framerate
    total_frames = int(duration_sec * fps)

    sample_idx = np.linspace(0, len(env) - 1, total_frames).astype(int)
    env_frames = env[sample_idx]

    cum = np.cumsum(env_frames)
    cum = cum / cum[-1]

    phi_audio = cum * (2 * np.pi * cycles)

    return phi_audio, total_frames, duration_sec


def load_text_phase(transcript_path, total_frames, cycles=TEXT_CYCLES):
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            _ = f.read().strip()
    except FileNotFoundError:
        pass

    t = np.linspace(0.0, 1.0, total_frames, endpoint=False)
    phi_text = t * (2 * np.pi * cycles)
    return phi_text


# ============================================
# TORUS GEOMETRY (Мёбиус-ТОР)
# ============================================

def torus_xyz(phi, theta):
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.array([x, y, z])


def rotate_z(vec, angle):
    x, y, z = vec
    c = np.cos(angle)
    s = np.sin(angle)
    x_new = x * c - y * s
    y_new = x * s + y * c
    return np.array([x_new, y_new, z])


# ============================================
# SAVE PNG WITH TEXT
# ============================================

def save_png_with_text(fig, filename, phase_idx, phi_text_value):
    fig.canvas.draw()

    buf = fig.canvas.renderer.buffer_rgba()
    img = np.asarray(buf, dtype=np.uint8)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    text = f"phase: {phase_idx}    phi_text: {phi_text_value:.3f} rad"
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    pil.save(filename)


# ============================================
# PRECOMPUTE PHASES
# ============================================

phi_audio_seq, TOTAL_FRAMES, AUDIO_DURATION = load_audio_phase(
    AUDIO_PATH, FPS, cycles=AUDIO_CYCLES
)
phi_text_seq = load_text_phase(TRANSCRIPT_PATH, TOTAL_FRAMES, cycles=TEXT_CYCLES)


# ============================================
# FIGURE SETUP
# ============================================

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1.4, 1.4)
ax.set_facecolor("black")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

spiral_points = []
spiral_colors = []

current_color = COLORS[0]
current_phase = 0

phi_text = 0.0
phi_audio = 0.0

main_phi = 0.0

follower_pos = torus_xyz(0.0, 0.0)


# ============================================
# ANIMATION UPDATE (READER)
# ============================================

def update(frame):
    global phi_text, phi_audio
    global follower_pos
    global spiral_points, spiral_colors
    global current_color, current_phase
    global main_phi

    phi_text_old = phi_text
    phi_text = float(phi_text_seq[frame])
    phi_audio = float(phi_audio_seq[frame])

    raw_delta = (phi_audio - phi_text) % (2 * np.pi)
    if raw_delta > np.pi:
        raw_delta -= 2 * np.pi
    delta_phi = raw_delta

    main_phi += MAIN_AXIS_GAIN * delta_phi

    # Мёбиус-тор: главный и боковой угол
    theta_mobius = phi_audio + 0.5 * phi_text

    base_pos = torus_xyz(phi_text, theta_mobius)
    rotated_target = rotate_z(base_pos, main_phi)

    follower_pos[:] = follower_pos + FOLLOWER_SMOOTH * (rotated_target - follower_pos)

    spiral_points.append(follower_pos.copy())
    spiral_colors.append(current_color)

    crossed = int(phi_text // CYCLE_2PI) > int(phi_text_old // CYCLE_2PI)

    ax.cla()
    ax.set_facecolor("black")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1.4, 1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # только траектория
    if len(spiral_points) > 1:
        start = 0
        for i in range(1, len(spiral_points)):
            if spiral_colors[i] != spiral_colors[i - 1]:
                seg = spiral_points[start:i]
                xs, ys, zs = zip(*seg)
                ax.plot(xs, ys, zs,
                        color=spiral_colors[i - 1],
                        linewidth=LINE_WIDTH)
                start = i

        seg = spiral_points[start:]
        xs, ys, zs = zip(*seg)
        ax.plot(xs, ys, zs,
                color=spiral_colors[-1],
                linewidth=LINE_WIDTH)

    ax.scatter(
        follower_pos[0],
        follower_pos[1],
        follower_pos[2],
        s=25,
        color="white",
        alpha=0.95
    )

    if crossed and SAVE_FRAMES:
        filename = f"{FRAMES_DIR}/spiral_{current_phase:02d}.png"
        save_png_with_text(fig, filename, current_phase, phi_text)
        current_phase += 1
        current_color = COLORS[current_phase % len(COLORS)]

    return []


# ============================================
# RUN ANIMATION
# ============================================

ani = animation.FuncAnimation(
    fig, update,
    frames=TOTAL_FRAMES,
    interval=1000 / FPS,
    blit=False
)

plt.show()
