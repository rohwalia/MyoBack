import gym
import os
import sys
sys.path.append('../')
sys.path.append('.')

from envs.myo.myobase.back_v0 import BackEnvV0
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import skvideo
import skvideo.io
import os
import cv2
import random
from tqdm.auto import tqdm
import warnings

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")

nb_seed = 1

torso = False
movie = True
path = './'

env_name = 'myoStandingBack-v0'

model_num = "baseline"
model = PPO.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name + '/' + model_num +
                 r'/best_model')



env = BackEnvV0(
                model_path='../myo_sim/back/myoback_v2.0.xml',
                target_jnt_range={
                    'LB_wrapjnt_t1': (0, 0), 'LB_wrapjnt_t2': (0, 0), 'LB_wrapjnt_r3': (0, 0),
                    'Abs_t1': (0, 0), 'Abs_t2': (0, 0), 'Abs_r3': (0, 0),
                    'flex_extension': (0, 0), 'lat_bending': (-0.436, 0.436), 'axial_rotation': (0, 0),
                    'L4_L5_FE': (0, 0), 'L4_L5_LB': (0, 0), 'L4_L5_AR': (0, 0),
                    'L3_L4_FE': (0, 0), 'L3_L4_LB': (0, 0), 'L3_L4_AR': (0, 0),
                    'L2_L3_FE': (0, 0), 'L2_L3_LB': (0, 0), 'L2_L3_AR': (0, 0),
                    'L1_L2_FE': (0, 0), 'L1_L2_LB': (0, 0), 'L1_L2_AR': (0, 0),
                },
                normalize_act=True,
                frame_skip=5,
                max_episode_steps=200
                )

env.reset()

random.seed() 

frames = []
view = 'side'
all_rewards = []
for _ in tqdm(range(2)):
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    while (not done) and (step < 500):
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1] 
          action, _ = model.predict(obs, deterministic= False)
          obs, reward, done, info, _ = env.step(action)
          ep_rewards.append(reward)
          if movie:
                  geom_1_indices = np.where(env.sim.model.geom_group == 1)
                  env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  frame = env.sim.renderer.render_offscreen(width= 640, height=480,camera_id=0)
                  frame = np.flipud(frame)
                  frames.append(frame[::-1,:,:])
          step += 1
    all_rewards.append(np.sum(ep_rewards))
print(f"Average reward: {np.mean(all_rewards)}")


if movie:
    os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'200'} , outputdict={"-pix_fmt": "yuv420p"})
	
