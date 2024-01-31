import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1_fw.go1_fw_config import config_go1_fw
from go1_gym.envs.go1_fw.velocity_tracking import VelocityTrackingSkatingEnv

from tqdm import tqdm


def load_rsl_policy(env, logdir, device = "cuda:0"):
    from rsl_rl.modules import ActorCritic
    from rsl_rl.algorithms import PPO
    train_cfg_dict = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 
                      'init_member_classes': {}, 
                      'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 
                      'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'roller_skating', 'load_run': -1, 'max_iterations': 600, 'num_steps_per_env': 24, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 
                      'runner_class_name': 'OnPolicyRunner', 
                      'seed': 1}
    num_critic_obs = env.num_obs
    actor_critic = ActorCritic(env.num_obs, num_critic_obs, env.num_actions, **train_cfg_dict['policy']).to(device)
    alg = PPO(actor_critic, device, train_cfg_dict["algorithm"])
    cfg = train_cfg_dict["runner"]
    alg.init_storage(env.num_envs, cfg["num_steps_per_env"], [env.num_obs], [env.num_privileged_obs], [env.num_actions])
    alg.actor_critic.eval()
    alg.actor_critic.to(device)
    policy = alg.actor_critic.act_inference
    return policy

def load_wtw_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(label, headless = False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]
    
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingSkatingEnv(sim_device="cuda:0", headless=False, cfg = Cfg)
    env = HistoryWrapper(env)
    return env

def play_go1(headless=True):
    from ml_logger import logger
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    

def test_rsl_model():
    # logdir = glob.glob(f"../runs/{label}/*")
    label = "gait-conditioned-agility/pretrain-v0/train"
    logdir = "../runs/skating/model_600.pt"
    env = load_env(label)
    policy = load_rsl_policy(env = env, logdir=logdir)
    print(policy)

if __name__ == "__main__":
    test_rsl_model()