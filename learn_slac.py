import numpy as np
import gym
import torch
from task import PendulumP, PendulumV, TaskT, ContinuousCartPoleEnv, CartPoleP, CartPoleV
import time, os, argparse, warnings
import scipy.io as sio
from copy import deepcopy
from slac import SLAC
try:
    from task import RsHopperV, RsHopperP, RsAntV, RsAntP, RsHalfCheetahP, RsHalfCheetahV, RsHumanoidP, RsHumanoidV, RsWalker2dP, RsWalker2dV
except:
    pass
try:
    from task import HopperV, HopperP
except:
    pass


def test_performance(agent_test, env_test, action_filter, times=5):

    EpiTestRet = 0

    for _ in range(times):

        # reset each episode
        sp_seq = np.zeros([seq_len, env.observation_space.shape[0]+1])
        s = env_test.reset()
        sp_seq[-1, :-1] = s
        sp_seq[-1, -1] = 0.0 # reward padding

        a = agent.select(sp_seq)
        for _ in range(10000):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            sp_seq[:-1] = deepcopy(sp_seq[1:])
            sp_seq[-1, :-1] = deepcopy(sp)
            sp_seq[-1, -1] = r
            a = agent_test.select(sp_seq, action_return='normal')  # for evaluating performance
            EpiTestRet += r
            if done:
                break

    EpiTestRet_mean = 0

    for _ in range(times):

        # reset each episode
        sp_seq = np.zeros([seq_len, env.observation_space.shape[0]+1])
        s = env_test.reset()
        sp_seq[-1, :-1] = s
        sp_seq[-1, -1] = 0.0 # reward padding

        a = agent.select(sp_seq)
        for _ in range(10000):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            sp_seq[:-1] = deepcopy(sp_seq[1:])
            sp_seq[-1, :-1] = deepcopy(sp)
            sp_seq[-1, -1] = r
            a = agent_test.select(sp_seq, action_return='mean')  # use tanh(mu_a) for evaluating performance
            EpiTestRet_mean += r
            if done:
                break

    return EpiTestRet / times, EpiTestRet_mean / times


savepath = './data_slac/'

if os.path.exists(savepath):
    warnings.warn('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)


# ================ Hyper-parameters =============
beta_h = 'auto_1.0'  # entropy coefficient of SAC ('\alpha' in the SAC paper)
batch_size = 32
seq_len = 8
gamma = 0.99
sigx = 'auto'  # sigma of output prediction, sigx = 0.33333 in the original implementation (for pixel observations)

step_start_rl = 1000  # step to start reinforcement learning (SAC)
step_start_st = 1000  # step to start learning the state transition model

train_step_rl = 1
train_freq_rl = 1. / train_step_rl
train_step_st = 1
train_freq_st = 1. / train_step_rl

# ============== Task-specific parameters =============

import roboschool
env = gym.make("RoboschoolHopper-v1")
env_test = gym.make("RoboschoolHopper-v1")
task_name = "rshopper"

action_filter = lambda a: a.reshape([-1])

max_steps = 1000
est_min_steps = 5

max_all_steps = 1000000
step_perf_eval = 5000


#===================== initialize ==================
max_episodes = int(max_all_steps / est_min_steps) + 1  # for replay buffer

agent = SLAC(input_size=env.observation_space.shape[0] + 1,
             action_size=env.action_space.shape[0],
             seq_len=seq_len,
             beta_h=beta_h,
             sigx=sigx)

agent_test = SLAC(input_size=env.observation_space.shape[0] + 1,
                  action_size=env.action_space.shape[0],
                  seq_len=seq_len,
                  beta_h=beta_h,
                  sigx=sigx)

S_buffer = np.zeros([max_episodes, max_steps+1, env.observation_space.shape[0]], dtype=np.float32)
A_buffer = np.zeros([max_episodes, max_steps, env.action_space.shape[0]], dtype=np.float32)
R_buffer = np.zeros([max_episodes, max_steps], dtype=np.float32)
D_buffer = np.zeros([max_episodes, max_steps], dtype=np.float32)  # done
V_buffer = np.zeros([max_episodes, max_steps], dtype=np.float32)  # whether a step is valid, value: 1 (compute gradient at this step) or 0 (stop gradient at this step)

performance_wrt_step = []
performance_mean_action_wrt_step = []
global_steps = []

episode = 0
global_step = 0

# ============ learning part ===============

while global_step < max_all_steps:

    sp_seq = np.zeros([seq_len, env.observation_space.shape[0] + 1])  # SLAC uses a sequence of observations for the input of the actor
    s = env.reset()
    S_buffer[episode, 0] = s.reshape([-1])
    sp_seq[-1, :-1] = s
    sp_seq[-1, -1] = 0.0  # reward padding

    a = agent.select(sp_seq)

    for t in range(max_steps):

        if global_step == max_all_steps:
            break

        if np.any(np.isnan(a)):
            raise ValueError

        sp, r, done, _ = env.step(action_filter(a))

        sp_seq[:-1] = deepcopy(sp_seq[1:])
        sp_seq[-1, :-1] = deepcopy(sp)
        sp_seq[-1, -1] = r

        A_buffer[episode, t] = a
        S_buffer[episode, t + 1] = sp.reshape([-1])
        R_buffer[episode, t] = r
        D_buffer[episode, t] = 1 if done else 0
        V_buffer[episode, t] = 1

        a = agent.select(sp_seq)

        global_step += 1
        s = deepcopy(sp)

        if global_step > step_start_st and np.random.rand() < train_freq_st:
            # training the latent variable model
            for _ in range(max(1, int(train_freq_st))):
                weights = np.sum(V_buffer[:episode], axis=-1) + 2 * seq_len - 2
                sample_es = np.random.choice(episode, batch_size, p=weights/weights.sum())
                #  sampling with such weights so that every step can be sampled with the same probability

                SP = S_buffer[sample_es, 1:].reshape([batch_size, -1, env.observation_space.shape[0]])
                A = A_buffer[sample_es].reshape([batch_size, -1, env.action_space.shape[0]])
                R = R_buffer[sample_es].reshape([batch_size, -1, 1])
                V = V_buffer[sample_es].reshape([batch_size, -1, 1])

                agent.train_st(x_obs=np.concatenate((SP, R), axis=-1), a_obs=A, r_obs=R, validity=V)

        if global_step > step_start_rl and np.random.rand() < train_freq_rl:
            # training the RL controller
            for _ in range(max(1, int(train_freq_rl))):
                weights = np.sum(V_buffer[:episode], axis=-1) + 2 * seq_len - 2
                sample_es = np.random.choice(episode, batch_size, p=weights/weights.sum())
                #  sampling with such weights so that every step can be sampled with the same probability

                SP = S_buffer[sample_es, 1:].reshape([batch_size, -1, env.observation_space.shape[0]])
                S0 = S_buffer[sample_es, 0].reshape([batch_size, env.observation_space.shape[0]])
                A = A_buffer[sample_es].reshape([batch_size, -1, env.action_space.shape[0]])
                R = R_buffer[sample_es].reshape([batch_size, -1, 1])
                D = D_buffer[sample_es].reshape([batch_size, -1, 1])
                V = V_buffer[sample_es].reshape([batch_size, -1, 1])

                agent.train_rl_sac(x_obs=np.concatenate((SP, R), axis=-1), s_0=S0,
                                   a_obs=A, r_obs=R, d_obs=D, validity=V, gamma=0.99)

        if global_step % step_perf_eval == 0:
            # evaluate performance
            agent_test.load_state_dict(agent.state_dict())  # update agent_test

            EpiTestRet, EpiTestRet_mean = test_performance(agent_test, env_test, action_filter, times=5)
            performance_wrt_step.append(EpiTestRet)
            performance_mean_action_wrt_step.append(EpiTestRet_mean)
            global_steps.append(global_step)
            warnings.warn(task_name + ": global step: {}, : steps {}, test return {}".format(
                global_step, t, EpiTestRet))

        if done:
            break

    print(task_name + " -- episode {} : steps {}, mean reward {}".format(episode, t, np.mean(R_buffer[episode])))
    episode += 1

performance_wrt_step = np.reshape(performance_wrt_step, [-1]).astype(np.float64)
performance_mean_action_wrt_step_array = np.reshape(performance_mean_action_wrt_step, [-1]).astype(np.float64)
global_steps = np.reshape(global_steps, [-1]).astype(np.float64)

data = {"seq_len": seq_len,
        "sigx": sigx,
        "beta_h": beta_h,
        "gamma": gamma,
        "codes": code_strs,
        "max_steps": max_steps,
        "max_episodes": max_episodes,
        "step_start_st": step_start_st,
        "step_start_rl": step_start_rl,
        "batch_size": batch_size,
        "train_step_rl": train_step_rl,
        "train_step_st": train_step_st,
        "R": np.sum(R_buffer, axis=-1).astype(np.float64),
        "steps": np.sum(V_buffer, axis=-1).astype(np.float64),
        "performance_wrt_step": performance_wrt_step,
        "performance_mean_action_wrt_step": performance_mean_action_wrt_step_array,
        "global_steps": global_steps}

sio.savemat(savepath + task_name + "_slac_{}.mat", data)  # save data
torch.save(agent, savepath + task_name + "_slac_{}.model")  # save model

