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

mu_c = [0.15, 0.20, 0.12]     
sigma_c = [0.015, 0.02, 0.012] 

R_kernel = 13          
x = np.arange(-SIZE//2, SIZE//2)
X, Y = np.meshgrid(x, x)

# [수정된 핵심 파트: 비등방성(Anisotropic) 커널 적용]
# 완벽한 원형(등방성)을 깨부수고 방향성을 부여함
theta = np.arctan2(Y, X)
# X, Y축의 비율을 다르게 하여 타원형으로 만듦
D = np.sqrt((X * 1.2)**2 + (Y * 0.8)**2) / R_kernel
K_base = np.exp(-(D - 0.5)**2 / 0.15) * (D < 1)

# 각도(theta)에 따라 3방향으로 뻗어나가는 비대칭성 추가
K = K_base * (1 + 0.4 * np.cos(3 * theta)) 
K = np.maximum(K, 0) # 음수 값 방지
K = K / np.sum(K)
K_fft = fft2(np.fft.ifftshift(K))

W = np.array([
    [1.0,  0.4, -0.9], 
    [-0.9, 1.0,  0.4], 
    [0.4, -0.9,  1.0]  
])

world = np.zeros((SIZE, SIZE, 3))
species = np.random.randint(0, NUM_SPECIES, NUM_AGENTS)

positions = np.random.rand(NUM_AGENTS, 2) * SIZE
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
    global positions, velocities, world
    
    agent_max_speeds = max_speeds[species][:, None]
    agent_noises = noise_levels[species][:, None]
    
    angles = np.random.randn(NUM_AGENTS, 2) * agent_noises
    velocities += angles
    
    speed = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = np.where(speed > agent_max_speeds, velocities / speed * agent_max_speeds, velocities)
    positions += velocities
    
    for i in range(NUM_AGENTS):
        for j in range(2):
            if positions[i, j] < 0 or positions[i, j] >= SIZE:
                velocities[i, j] *= -1
                positions[i, j] = np.clip(positions[i, j], 0, SIZE-1)
                
    for i in range(NUM_AGENTS):
        x_pos, y_pos = int(positions[i, 0]), int(positions[i, 1])
        world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] = np.clip(world[x_pos-1:x_pos+2, y_pos-1:y_pos+2] + colors[i] * 0.3, 0, 1)
        
    potentials = [np.real(ifft2(K_fft * fft2(world[:, :, c]))) for c in range(3)]
    
    for c in range(3):
        u = sum(W[c, i] * potentials[i] for i in range(3))
        growth = 2 * np.exp(-((u - mu_c[c])**2) / (2 * sigma_c[c]**2)) - DECAY
        world[:, :, c] = np.clip(world[:, :, c] + dt * growth, 0, 1)
        
    img.set_data(world)
    return [img]

ani = animation.FuncAnimation(fig, update, frames=500, interval=40, blit=True)
plt.show()