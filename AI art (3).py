import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2

# 1. 기본 및 환경 설정
SIZE = 150
MAX_AGENTS = 300  # 환경이 수용 가능한 최대 개체 수
T = 10
dt = 1.0 / T
DECAY = 0.55

mu_c = [0.15, 0.15, 0.15]
sigma_c = [0.012, 0.012, 0.012]

R_kernel = 13
x = np.arange(-SIZE//2, SIZE//2)
X, Y = np.meshgrid(x, x)

# 비등방성 커널 (환경의 기하학적 형태 부여)
theta = np.arctan2(Y, X)
D = np.sqrt((X * 1.2)**2 + (Y * 0.8)**2) / R_kernel
K_aniso = np.exp(-(D - 0.5)**2 / 0.15) * (D < 1)
K_aniso = K_aniso * (1 + 0.8 * np.cos(2 * theta))
K_aniso = np.maximum(K_aniso, 0)
K_aniso = K_aniso / np.sum(K_aniso)
K_aniso_fft = fft2(np.fft.ifftshift(K_aniso))

W = np.array([
    [1.0,  0.4, -0.9], 
    [-0.9, 1.0,  0.4], 
    [0.4, -0.9,  1.0]  
])

world = np.zeros((SIZE, SIZE, 3))
agent_world = np.zeros((SIZE, SIZE, 3))

# 2. 개체(Agent) 특성 배열 초기화
alive = np.zeros(MAX_AGENTS, dtype=bool)
positions = np.zeros((MAX_AGENTS, 2))
prev_positions = np.zeros((MAX_AGENTS, 2))
velocities = np.zeros((MAX_AGENTS, 2))

colors = np.zeros((MAX_AGENTS, 3))
max_speeds = np.zeros(MAX_AGENTS)
noise_levels = np.zeros(MAX_AGENTS)

energy = np.zeros(MAX_AGENTS)       
generation = np.zeros(MAX_AGENTS)   

# [수정됨] 빅뱅 시작점 (초기 조상 설정)
INITIAL_AGENTS = 10
alive[:INITIAL_AGENTS] = True
positions[:INITIAL_AGENTS] = SIZE / 2.0  
prev_positions[:INITIAL_AGENTS] = SIZE / 2.0

angles = np.linspace(0, 2 * np.pi, INITIAL_AGENTS, endpoint=False)
velocities[:INITIAL_AGENTS, 0] = np.cos(angles) * 8.0
velocities[:INITIAL_AGENTS, 1] = np.sin(angles) * 8.0

colors[:INITIAL_AGENTS] = np.random.rand(INITIAL_AGENTS, 3)
max_speeds[:INITIAL_AGENTS] = np.random.uniform(2.0, 5.0, INITIAL_AGENTS)
noise_levels[:INITIAL_AGENTS] = np.random.uniform(0.1, 0.5, INITIAL_AGENTS)

# [수정됨] 초기 개체들에게 환경이 자랄 때까지 버틸 수 있는 넉넉한 에너지(1.2) 부여
energy[:INITIAL_AGENTS] = 1.2  
generation[:INITIAL_AGENTS] = 1

fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')
img = ax.imshow(world, interpolation='bilinear')

def update(frame):
    global positions, prev_positions, velocities, world, agent_world
    global alive, colors, max_speeds, noise_levels, energy, generation
    
    # 1. 개체 이동 및 환경 상호작용
    for i in range(MAX_AGENTS):
        if not alive[i]:
            continue
            
        angle_noise = np.random.randn(2) * noise_levels[i]
        velocities[i] += angle_noise
        
        speed = np.linalg.norm(velocities[i])
        if speed > max_speeds[i]:
            velocities[i] = (velocities[i] / speed) * max_speeds[i]
            
        prev_positions[i] = positions[i]
        positions[i] += velocities[i]
        
        for j in range(2):
            if positions[i, j] < 1 or positions[i, j] >= SIZE - 1:
                velocities[i, j] *= -1
                positions[i, j] = np.clip(positions[i, j], 1, SIZE-2)
                
        x_pos, y_pos = int(positions[i, 0]), int(positions[i, 1])
        
        # [수정됨] 대사량: 기초 대사량을 낮춰 덜 굶주리게 함 (0.1 -> 0.05)
        energy[i] -= dt * 0.05 
        
        # [수정됨] 섭식: 영양분 흡수 효율을 대폭 증가시킴 (0.05 -> 0.15)
        pref_channel = np.argmax(colors[i])
        energy[i] += world[x_pos, y_pos, pref_channel] * 0.15
        
        if energy[i] <= 0:
            alive[i] = False
            continue
            
    # [수정됨] 번식 시스템: 번식 에너지 임계값을 낮춤 (1.5 -> 1.4)
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
        
        # 유전자 복제 및 돌연변이
        colors[child_idx] = np.clip(colors[parent_idx] + np.random.randn(3) * 0.1, 0, 1)
        max_speeds[child_idx] = np.clip(max_speeds[parent_idx] + np.random.randn() * 0.5, 1.0, 6.0)
        noise_levels[child_idx] = np.clip(noise_levels[parent_idx] + np.random.randn() * 0.05, 0.0, 1.0)
        
        generation[child_idx] = generation[parent_idx] + 1
        
        # [수정됨] 에너지 분배 방식 변경
        # 부모는 번식에 에너지를 일정량(0.5)만 소모하고 살아남음
        energy[parent_idx] -= 0.5  
        # 자식은 태어날 때 생존 가능한 초기 에너지(0.8)를 가지고 태어남
        energy[child_idx] = 0.8   

    # 3. 환경(World) 업데이트 
    agent_world *= 0.85
    
    for i in range(MAX_AGENTS):
        if not alive[i]:
            continue
            
        steps = int(max(1, np.linalg.norm(positions[i] - prev_positions[i]) * 2))
        for step in range(steps + 1):
            alpha = step / max(1, steps)
            interp_pos = prev_positions[i] * (1 - alpha) + positions[i] * alpha
            x_pos, y_pos = np.clip(int(interp_pos[0]), 1, SIZE-2), np.clip(int(interp_pos[1]), 1, SIZE-2)
            
            agent_world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] = np.clip(
                agent_world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] + colors[i] * (0.5 * alpha), 0, 1
            )
            
    combined_world = np.clip(world + agent_world, 0, 1)
    potentials = [np.real(ifft2(K_aniso_fft * fft2(combined_world[:, :, c]))) for c in range(3)]
    
    for c in range(3):
        u = sum(W[c, i] * potentials[i] for i in range(3))
        local_mu = mu_c[c] - 0.05 * agent_world[:, :, c] + 0.03 * agent_world[:, :, (c+1)%3]
        growth = 2 * np.exp(-((u - local_mu)**2) / (2 * sigma_c[c]**2)) - DECAY
        world[:, :, c] = np.clip(world[:, :, c] + dt * growth, 0, 1)
        
    display_world = np.clip(world + agent_world, 0, 1)
    img.set_data(display_world)
    
    return [img]

ani = animation.FuncAnimation(fig, update, frames=500, interval=40, blit=True)
plt.show()