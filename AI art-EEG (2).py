import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2
import mne
from mne.datasets import eegbci

# ---------------------------------------------------------
# 0. 오픈소스 EEG 데이터 로드 및 전처리
# ---------------------------------------------------------
print("뇌파 데이터 로드 중...")
raw_fnames = eegbci.load_data(1, [1]) 
raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)

data, _ = raw.get_data(return_times=True)
eeg_raw_signal = data[0, :] 

eeg_min = np.min(eeg_raw_signal)
eeg_max = np.max(eeg_raw_signal)
eeg_signal = (eeg_raw_signal - eeg_min) / (eeg_max - eeg_min)
eeg_length = len(eeg_signal)

# ---------------------------------------------------------
# 1. 기본 및 환경 설정
# ---------------------------------------------------------
SIZE = 150
MAX_AGENTS = 300  
T = 10
dt = 1.0 / T

mu_c = [0.15, 0.15, 0.15]
sigma_c = [0.012, 0.012, 0.012]

x = np.arange(-SIZE//2, SIZE//2)
X, Y = np.meshgrid(x, x)

# ---------------------------------------------------------
# [수정됨] 상호작용 행렬 (적대적 -> 공생적)
# 마이너스(-) 값을 모두 없애고 서로 부드럽게 섞이도록 유도
# ---------------------------------------------------------
W = np.array([
    [1.0,  0.6,  0.6], 
    [0.6,  1.0,  0.6], 
    [0.6,  0.6,  1.0]  
])

world = np.zeros((SIZE, SIZE, 3))
agent_world = np.zeros((SIZE, SIZE, 3))

# ---------------------------------------------------------
# 2. 개체(Agent) 특성 배열 초기화
# ---------------------------------------------------------
alive = np.zeros(MAX_AGENTS, dtype=bool)
positions = np.zeros((MAX_AGENTS, 2))
prev_positions = np.zeros((MAX_AGENTS, 2))
velocities = np.zeros((MAX_AGENTS, 2))

colors = np.zeros((MAX_AGENTS, 3))
max_speeds = np.zeros(MAX_AGENTS)
base_noise_levels = np.zeros(MAX_AGENTS)

energy = np.zeros(MAX_AGENTS)       
generation = np.zeros(MAX_AGENTS)   

INITIAL_AGENTS = 10
alive[:INITIAL_AGENTS] = True
positions[:INITIAL_AGENTS] = SIZE / 2.0  
prev_positions[:INITIAL_AGENTS] = SIZE / 2.0

angles = np.linspace(0, 2 * np.pi, INITIAL_AGENTS, endpoint=False)
velocities[:INITIAL_AGENTS, 0] = np.cos(angles) * 8.0
velocities[:INITIAL_AGENTS, 1] = np.sin(angles) * 8.0

colors[:INITIAL_AGENTS] = np.random.rand(INITIAL_AGENTS, 3)
max_speeds[:INITIAL_AGENTS] = np.random.uniform(2.0, 5.0, INITIAL_AGENTS)
base_noise_levels[:INITIAL_AGENTS] = np.random.uniform(0.1, 0.5, INITIAL_AGENTS)

energy[:INITIAL_AGENTS] = 1.2  
generation[:INITIAL_AGENTS] = 1

fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')
img = ax.imshow(world, interpolation='bilinear')

# ---------------------------------------------------------
# 3. 업데이트 함수
# ---------------------------------------------------------
def update(frame):
    global positions, prev_positions, velocities, world, agent_world
    global alive, colors, max_speeds, base_noise_levels, energy, generation
    
    current_eeg_val = eeg_signal[frame % eeg_length]
    
    window_size = 128
    start_idx = frame % max(1, eeg_length - window_size)
    eeg_window = eeg_signal[start_idx : start_idx + window_size]
    fft_vals = np.abs(np.fft.rfft(eeg_window))
    
    delta_p = np.mean(fft_vals[1:3])   
    theta_p = np.mean(fft_vals[3:6])   
    alpha_p = np.mean(fft_vals[6:10])  
    beta_p  = np.mean(fft_vals[10:20]) 
    gamma_p = np.mean(fft_vals[20:40]) 
    
    p_max = max(1e-5, np.max(fft_vals))
    delta, theta, alpha, beta, gamma = [np.clip(p / p_max, 0, 1) for p in [delta_p, theta_p, alpha_p, beta_p, gamma_p]]
    
    current_R = max(3.0, 3.0 + (current_eeg_val * 15.0))
    D_dynamic = np.sqrt(X**2 + Y**2) / current_R
    
    K_dynamic = np.exp(-(D_dynamic)**2 / 0.4) * (D_dynamic < 1)
    K_dynamic = np.maximum(K_dynamic, 0)
    
    k_sum = np.sum(K_dynamic)
    if k_sum > 0:
        K_dynamic = K_dynamic / k_sum
    else:
        K_dynamic[SIZE//2, SIZE//2] = 1.0
        
    K_dynamic_fft = fft2(np.fft.ifftshift(K_dynamic))
    
    dynamic_decay = 0.8 - (delta * 0.6)
    
    # 1. 개체 이동 및 상호작용
    for i in range(MAX_AGENTS):
        if not alive[i]:
            continue
            
        current_noise = base_noise_levels[i] + (gamma * 2.0)
        angle_noise = np.random.randn(2) * current_noise
        velocities[i] += angle_noise
        
        colors[i, 0] = np.clip(colors[i, 0] + (beta * 0.05) - 0.02, 0, 1)
        colors[i, 1] = np.clip(colors[i, 1] + (alpha * 0.05) - 0.02, 0, 1)
        colors[i, 2] = np.clip(colors[i, 2] + (theta * 0.05) - 0.02, 0, 1)
        
        current_max_speed = max_speeds[i] + (beta * 4.0)
        
        speed = np.linalg.norm(velocities[i])
        if speed > current_max_speed:
            velocities[i] = (velocities[i] / speed) * current_max_speed
            
        prev_positions[i] = positions[i]
        positions[i] += velocities[i]
        
        for j in range(2):
            if positions[i, j] < 1 or positions[i, j] >= SIZE - 1:
                velocities[i, j] *= -1
                positions[i, j] = np.clip(positions[i, j], 1, SIZE-2)
                
        x_pos, y_pos = int(positions[i, 0]), int(positions[i, 1])
        
        current_metabolism = 0.05 - (alpha * 0.03)
        energy[i] -= dt * max(0.01, current_metabolism)
        
        # 내적을 이용해 여러 색깔의 영양분을 골고루 섭취
        absorption = np.dot(colors[i], world[x_pos, y_pos])
        energy[i] += absorption * 0.08  
        
        if energy[i] <= 0:
            alive[i] = False
            continue
            
    # 번식 시스템
    reproducing_indices = np.where(alive & (energy > 1.4))[0]
    dead_indices = np.where(~alive)[0]
    
    for parent_idx in reproducing_indices:
        if len(dead_indices) == 0:
            break 
            
        child_idx = dead_indices[0]
        dead_indices = dead_indices[1:] 
        
        alive[child_idx] = True
        positions[child_idx] = positions[parent_idx]
        prev_positions[child_idx] = positions[parent_idx]
        velocities[child_idx] = -velocities[parent_idx] * 0.5
        
        mutation_rate = 0.1 + (gamma * 0.5)
        colors[child_idx] = np.clip(colors[parent_idx] + np.random.randn(3) * mutation_rate, 0, 1)
        max_speeds[child_idx] = np.clip(max_speeds[parent_idx] + np.random.randn() * (0.5 + gamma), 1.0, 8.0)
        base_noise_levels[child_idx] = np.clip(base_noise_levels[parent_idx] + np.random.randn() * (0.05 + gamma * 0.2), 0.0, 1.0)
        
        generation[child_idx] = generation[parent_idx] + 1
        
        energy[parent_idx] -= 0.5  
        energy[child_idx] = 0.8   

    # 4. 환경(World) 업데이트 
    agent_world *= 0.85
    
    for i in range(MAX_AGENTS):
        if not alive[i]:
            continue
            
        steps = int(max(1, np.linalg.norm(positions[i] - prev_positions[i]) * 2))
        for step in range(steps + 1):
            alpha_step = step / max(1, steps)
            interp_pos = prev_positions[i] * (1 - alpha_step) + positions[i] * alpha_step
            x_pos, y_pos = np.clip(int(interp_pos[0]), 1, SIZE-2), np.clip(int(interp_pos[1]), 1, SIZE-2)
            
            agent_world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] = np.clip(
                agent_world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] + colors[i] * (0.5 * alpha_step), 0, 1
            )
            
    combined_world = np.clip(world + agent_world, 0, 1)
    
    potentials = [np.real(ifft2(K_dynamic_fft * fft2(combined_world[:, :, c]))) for c in range(3)]
    
    for c in range(3):
        u = sum(W[c, i] * potentials[i] for i in range(3))
        # [수정됨] 서로 밀어내는 잔여 로직 삭제. 오직 자기 자신의 밀도만 안정화시킴
        local_mu = mu_c[c] - 0.05 * agent_world[:, :, c] 
        
        growth = 2 * np.exp(-((u - local_mu)**2) / (2 * sigma_c[c]**2)) - dynamic_decay
        world[:, :, c] = np.clip(world[:, :, c] + dt * growth, 0, 1)
        
    display_world = np.clip(world + agent_world, 0, 1)
    img.set_data(display_world)
    
    return [img]

ani = animation.FuncAnimation(fig, update, frames=1000, interval=40, blit=True)
plt.show()