import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2

# 1. 기본 설정
SIZE = 150
NUM_AGENTS = 30
NUM_SPECIES = 4 
T = 10
dt = 1.0 / T

DECAY = 0.55 

# 파라미터 대칭화
mu_c = [0.15, 0.15, 0.15]     
# [수정] sigma_c를 낮춰서 둥글게 퍼지는 현상(점)을 억제하고 선명하게 깎음
sigma_c = [0.012, 0.012, 0.012] 

R_kernel = 13          
x = np.arange(-SIZE//2, SIZE//2)
X, Y = np.meshgrid(x, x)

# 2. 비등방성 커널: 선형(Stripe) 패턴을 강제하기 위해 2-fold 대칭 적용
theta = np.arctan2(Y, X)
D = np.sqrt((X * 1.2)**2 + (Y * 0.8)**2) / R_kernel

K_base = np.exp(-(D - 0.5)**2 / 0.15) * (D < 1)
# [수정] cos(2*theta)로 길쭉한 뱡향성을 강하게 부여하여 동그란 점 파괴
K_aniso = K_base * (1 + 0.8 * np.cos(2 * theta)) 
K_aniso = np.maximum(K_aniso, 0)
K_aniso = K_aniso / np.sum(K_aniso)
K_aniso_fft = fft2(np.fft.ifftshift(K_aniso))

W = np.array([
    [1.0,  0.4, -0.9], 
    [-0.9, 1.0,  0.4], 
    [0.4, -0.9,  1.0]  
])

# 3. 레이어 및 개체 상태 분리
world = np.zeros((SIZE, SIZE, 3))         
agent_world = np.zeros((SIZE, SIZE, 3))   

species = np.random.randint(0, NUM_SPECIES, NUM_AGENTS)
# [수정] 생존 여부를 판별하는 배열 추가
alive = np.ones(NUM_AGENTS, dtype=bool)

positions = np.random.rand(NUM_AGENTS, 2) * SIZE
prev_positions = np.copy(positions)
velocities = (np.random.rand(NUM_AGENTS, 2) - 0.5) * 4.0

base_colors = np.array([
    [1.0, 0.2, 0.2], 
    [0.2, 1.0, 0.2], 
    [0.2, 0.2, 1.0], 
    [1.0, 0.8, 0.2]  
])
max_speeds = np.array([5.0, 1.5, 3.0, 0.8])
noise_levels = np.array([0.05, 0.8, 0.3, 0.1])

colors = base_colors[species] + (np.random.rand(NUM_AGENTS, 3) * 0.2 - 0.1)
colors = np.clip(colors, 0, 1)

fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')

img = ax.imshow(world, interpolation='bilinear')

def update(frame):
    global positions, prev_positions, velocities, world, agent_world, alive
    
    agent_max_speeds = max_speeds[species][:, None]
    agent_noises = noise_levels[species][:, None]
    
    # 살아있는 개체만 각도 변경 및 이동
    angles = np.random.randn(NUM_AGENTS, 2) * agent_noises
    velocities = np.where(alive[:, None], velocities + angles, 0)
    
    speed = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = np.where(speed > agent_max_speeds, velocities / (speed + 1e-8) * agent_max_speeds, velocities)
    
    prev_positions[:] = positions[:]
    positions = np.where(alive[:, None], positions + velocities, positions)
    
    # [수정] 사망 조건 및 경계 처리
    for i in range(NUM_AGENTS):
        if not alive[i]:
            continue
            
        for j in range(2):
            if positions[i, j] < 0 or positions[i, j] >= SIZE:
                velocities[i, j] *= -1
                positions[i, j] = np.clip(positions[i, j], 0, SIZE-1)
                
        # 현재 위치의 색상 농도 확인
        x_pos, y_pos = int(np.clip(positions[i, 0], 0, SIZE-1)), int(np.clip(positions[i, 1], 0, SIZE-1))
        
        # 각 종마다 취약한 '독성 색상 채널' 할당 (자신의 종 번호에 따라 결정)
        poison_channel = (species[i] + 1) % 3 
        
        # 특정 색상(독)의 농도가 0.7 이상이면 개체 삭제 (사망)
        if world[x_pos, y_pos, poison_channel] > 0.7:
            alive[i] = False
                
    agent_world *= 0.85
    
    # 살아있는 개체만 꼬리 흔적 기록
    for i in range(NUM_AGENTS):
        if not alive[i]:
            continue
            
        steps = int(max(1, np.linalg.norm(positions[i] - prev_positions[i]) * 2))
        for step in range(steps + 1):
            alpha = step / max(1, steps)
            interp_pos = prev_positions[i] * (1 - alpha) + positions[i] * alpha
            x_pos, y_pos = int(interp_pos[0]), int(interp_pos[1])
            
            x_pos = np.clip(x_pos, 1, SIZE-2)
            y_pos = np.clip(y_pos, 1, SIZE-2)
            
            tail_weight = 0.5 * alpha 
            agent_world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] = np.clip(
                agent_world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] + colors[i] * tail_weight, 0, 1
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