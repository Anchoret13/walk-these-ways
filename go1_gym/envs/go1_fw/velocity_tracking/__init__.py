from isaacgym import gymtorch, gymutil, gymapi
from isaacgym.torch_utils import *
import os
import torch
from params_proto import Meta
from typing import Union, Dict

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.base.legged_robot_config import Cfg

class VelocityTrackingSkatingEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs = None, prone = None, deploy = False, 
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict = None, physics_engine = "SIM_PHYSX"):
        if num_envs is not None:
            cfg.env.num_envs = num_envs
        self.num_actions = 12
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = self.cfg.noise.add_noise
        # noise_scales = self.cfg.noise.scales
        noise_level = self.cfg.noise.noise_level
        

    def compute_observations(self):
        cmd = self.commands * self.commands_scale
        cmd = cmd[:3]
        self.obs_buf = torch.cat((
            self.projected_gravity,
            cmd,
            (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                     :self.num_actuated_dof]) * self.obs_scales.dof_pos,
            self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
            self.actions
        ), dim = -1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        


    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.render_gui()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs