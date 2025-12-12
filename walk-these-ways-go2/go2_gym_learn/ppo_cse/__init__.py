import time
from collections import deque
import copy
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

from go2_gym import MINI_GYM_ROOT_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go2_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 4000  # check for potential saves every this many iterations
    save_video_interval = 4000
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True


class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env

        # 新增：安全创建 TensorBoard writer（若不可用则禁用）
        try:
            tb_dir = os.path.join(MINI_GYM_ROOT_DIR, "logs", "tensorboard", time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            # counters for episode-based x-axis (increment per finished episode)
            self._tb_train_ep_count = 0
            self._tb_eval_ep_count = 0
        except Exception as e:
            print("Warning: TensorBoard SummaryWriter unavailable, disabling TB logging:", e)
            self.tb_writer = None

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)

        if RunnerArgs.resume:
            # try to load pretrained weights from a local resume_path first (prefer local),
            # otherwise fall back to ML_Logger remote loader
            try:
                loader = None
                if RunnerArgs.resume_path is not None and os.path.exists(RunnerArgs.resume_path):
                    # expect directory structure: <resume_path>/checkpoints/ac_weights_last.pt
                    local_chk = os.path.join(RunnerArgs.resume_path, "checkpoints", "ac_weights_last.pt")
                    if os.path.exists(local_chk):
                        print(f"Resuming actor_critic from local checkpoint: {local_chk}")
                        weights = torch.load(local_chk, map_location=self.device)
                        actor_critic.load_state_dict(state_dict=weights)
                    elif os.path.isfile(RunnerArgs.resume_path):
                        # resume_path might be a direct checkpoint file
                        print(f"Resuming actor_critic from local checkpoint file: {RunnerArgs.resume_path}")
                        weights = torch.load(RunnerArgs.resume_path, map_location=self.device)
                        actor_critic.load_state_dict(state_dict=weights)
                    else:
                        # fallback to remote loader
                        from ml_logger import ML_Logger
                        loader = ML_Logger(root="http://escher.csail.mit.edu:8080", prefix=RunnerArgs.resume_path)
                        weights = loader.load_torch("checkpoints/ac_weights_last.pt")
                        actor_critic.load_state_dict(state_dict=weights)
                else:
                    # no local path provided — try ML_Logger remote loader
                    from ml_logger import ML_Logger
                    loader = ML_Logger(root="http://escher.csail.mit.edu:8080", prefix=RunnerArgs.resume_path)
                    weights = loader.load_torch("checkpoints/ac_weights_last.pt")
                    actor_critic.load_state_dict(state_dict=weights)

                if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                    # load curriculum state: prefer a local pkl when resuming from local path,
                    # otherwise use the remote loader if available
                    distributions = None
                    try:
                        if RunnerArgs.resume_path is not None and os.path.exists(RunnerArgs.resume_path):
                            local_pkl = os.path.join(RunnerArgs.resume_path, "curriculum", "distribution.pkl")
                            if os.path.exists(local_pkl):
                                try:
                                    import pickle
                                    with open(local_pkl, "rb") as f:
                                        distributions = pickle.load(f)
                                        print(f"Loaded curriculum distributions from local pkl: {local_pkl}")                                        
                                except Exception:
                                    distributions = None
                        # if not loaded from local, try remote loader (if set)
                        if distributions is None and 'loader' in locals() and loader is not None:
                            try:
                                distributions = loader.load_pkl("curriculum/distribution.pkl")
                            except Exception:
                                distributions = None

                        if distributions:
                            distribution_last = distributions[-1]["distribution"]
                            for gait_id, gait_name in enumerate(self.env.category_names):
                                key = f"weights_{gait_name}"
                                if key in distribution_last:
                                    self.env.curricula[gait_id].weights = distribution_last[key]
                                    print(gait_name)
                    except Exception:
                        # be conservative: if anything goes wrong, skip curriculum restore
                        pass
            except Exception as e:
                print("Warning: could not resume from checkpoint:", e)

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        # helper: 把可能是 tensor 或 numpy 的值转成 Python float（失败返回 None）
        def _to_scalar(x):
            try:
                if torch.is_tensor(x):
                    return x.item()
                if isinstance(x, (np.ndarray, np.generic)):
                    return float(x)
                return float(x)
            except Exception:
                return None

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])
                        # 写入 TensorBoard: 将 infos['train/episode'] 中的标量逐个写入
                        if self.tb_writer is not None:
                            try:
                                for k, v in infos['train/episode'].items():
                                    if hasattr(v, "item"):
                                        val = float(v.item())
                                    else:
                                        val = float(v)
                                    self.tb_writer.add_scalar(f"train/episode/{k}", val, self._tb_train_ep_count)
                                self._tb_train_ep_count += 1
                            except Exception:
                                pass

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])
                        # 写入 TensorBoard: eval episode
                        if self.tb_writer is not None:
                            try:
                                for k, v in infos['eval/episode'].items():
                                    if hasattr(v, "item"):
                                        val = float(v.item())
                                    else:
                                        val = float(v)
                                    self.tb_writer.add_scalar(f"eval/episode/{k}", val, self._tb_eval_ep_count)
                                self._tb_eval_ep_count += 1
                            except Exception:
                                pass

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                if it % curriculum_dump_freq == 0:
                    logger.save_pkl({"iteration": it,
                                     **caches.slot_cache.get_summary(),
                                     **caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

                    if 'curriculum/distribution' in infos:
                        logger.save_pkl({"iteration": it,
                                         "distribution": distribution},
                                         path=f"curriculum/distribution.pkl", append=True)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            logger.store_metrics(
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            )

            # 写入 TensorBoard: 把主要训练指标以 iteration 为 x 轴记录
            if self.tb_writer is not None:
                try:
                    self.tb_writer.add_scalar('train/time_elapsed', _to_scalar(logger.since('start')), it)
                    self.tb_writer.add_scalar('train/time_iter', _to_scalar(logger.split('epoch')), it)
                    self.tb_writer.add_scalar('train/adaptation_loss', _to_scalar(mean_adaptation_module_loss), it)
                    self.tb_writer.add_scalar('train/mean_value_loss', _to_scalar(mean_value_loss), it)
                    self.tb_writer.add_scalar('train/mean_surrogate_loss', _to_scalar(mean_surrogate_loss), it)
                    self.tb_writer.add_scalar('train/mean_decoder_loss', _to_scalar(mean_decoder_loss), it)
                    self.tb_writer.add_scalar('train/mean_decoder_loss_student', _to_scalar(mean_decoder_loss_student), it)
                    self.tb_writer.add_scalar('train/mean_decoder_test_loss', _to_scalar(mean_decoder_test_loss), it)
                    self.tb_writer.add_scalar('train/mean_decoder_test_loss_student', _to_scalar(mean_decoder_test_loss_student), it)
                    self.tb_writer.add_scalar('train/mean_adaptation_module_test_loss', _to_scalar(mean_adaptation_module_test_loss), it)
                except Exception:
                    pass

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()

            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            self.current_learning_iteration += num_learning_iterations

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = './tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

        # 关闭 TensorBoard writer（如果存在）
        if hasattr(self, 'tb_writer') and self.tb_writer is not None:
            try:
                self.tb_writer.close()
            except Exception:
                pass


    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
