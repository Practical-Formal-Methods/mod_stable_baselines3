import copy
import time
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th


from mod_gym import gym

from mod_stable_baselines3.stable_baselines3.common.base_class import BaseAlgorithm
from mod_stable_baselines3.stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from mod_stable_baselines3.stable_baselines3.common.callbacks import BaseCallback
from mod_stable_baselines3.stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from mod_stable_baselines3.stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from mod_stable_baselines3.stable_baselines3.common.utils import obs_as_tensor, safe_mean, hellinger, get_partitions, tune_alpha, get_hashes, get_cov
from mod_stable_baselines3.stable_baselines3.common.vec_env import VecEnv


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        self.best_bugs = float('inf')
        self.best_rew_of_bb  = float('-inf')

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        self.setup_test()
        
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()


            if self.num_timesteps % (2048 * 50) == 0:
                self.test()
            
            # we put it here to keep guided and normal the same
            # if self.num_timesteps % (2048 * 200) == 0:
            #     self.game.env.seed(self.seed)
            #     avg_rew = self.game.eval(eval_budget=30)

            # if not self.train_type == "normal" and self.num_timesteps % (2048 * 200) == 0:
                
            #     fw = open(self.log_dir + "/info.log", "a")

            #     if self.env_iden == "car_racing":
            #         self.env.venv.last_avg_rew = avg_rew 
            #         if avg_rew > self.env.venv.guide_rew:
            #             self.env.venv.guide_prob = min(self.env.venv.guide_prob + self.guide_prob_inc, self.guide_prob_thold)
            #         else:
            #             self.env.venv.guide_prob = max(0, self.env.venv.guide_prob - self.guide_prob_inc)
            #         fw.write("New guide probability is %f.\n" % self.env.venv.guide_prob)
            #     else:
            #         self.env.last_avg_rew = avg_rew
            #         if avg_rew > self.env.guide_rew:
            #             self.env.guide_prob = min(self.env.guide_prob + self.guide_prob_inc, self.guide_prob_thold)
            #         else:
            #             self.env.guide_prob = max(0, self.env.guide_prob - self.guide_prob_inc)

            #         fw.write("New guide probability is %f.\n" % self.env.guide_prob)

            #     fw.close()


        callback.on_training_end()

        self.post_train()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


    def setup_test(self):
        from lunar.Mutator import LunarOracleMoonHeightMutator
        from lunar.EnvWrapper import Wrapper as LunarWrapper
        from bipedal.Mutator import BipedalEasyOracleMutator
        from bipedal.EnvWrapper import Wrapper as BipedalWrapper
        from car_racing.Mutator import CarRacingRoadMutator
        from car_racing.EnvWrapper import Wrapper as CarRacingWrapper

        if self.env_iden == "lunar":
            game = LunarWrapper(self.env_iden)
            game.env = self.env
            game.model = self.policy
            self.game = game 
            mutator = LunarOracleMoonHeightMutator(game)
        elif self.env_iden == "bipedal":
            game = BipedalWrapper(self.env_iden)
            game.env = self.env
            game.model = self.policy
            self.game = game 
            mutator = BipedalEasyOracleMutator(game)
        elif self.env_iden == "car_racing":
            game = CarRacingWrapper(self.env_iden)
            game.env = self.env
            game.model = self.policy
            self.game = game 
            mutator = CarRacingRoadMutator(game)

        rng = np.random.default_rng(self.seed)

        poolfile= open("%s/state_pool.p" % self.env_iden, 'rb')
        pool = pickle.load(poolfile)[1]
        self.pool = rng.choice(pool, self.explr_budget)
        self.testsuite = []  # defaultdict(list)
        for org_idx, org in enumerate(self.pool):
            org_state = org.hi_lvl_state
            # if org_idx % 2 == 0:
            rlx_state = mutator.mutate(org_state, rng, 'relax')
            self.testsuite.append([org_idx, rlx_state, 'rlx'])
            # else:
            unrlx_state = mutator.mutate(org_state, rng, 'unrelax')
            if unrlx_state is not None:
                self.testsuite.append([org_idx, unrlx_state, 'unrlx'])

        self.all_gstates_by_test = []
        self.all_nstates_by_test = []


    def test(self):
        
        if self.env_iden == "car_racing": self.env.venv.locked = True
        else: self.env.locked = True

        num_rlx_bugs, rlx_fp = 0, 0
        num_unrlx_bugs, unrlx_fp = 0, 0
        self.guiding_init_nnstates = list()
        cur_guiding_states = []
        guiding_st_idx = []
        for org_idx, mut_st, mut_type in self.testsuite:  # .items():
            org = self.pool[org_idx]
            
            org_llvl = self.env.reset(org.hi_lvl_state, org.rand_state)
            o_org = self.game.play(org_llvl)
            mut_llvl = self.env.reset(mut_st, org.rand_state)
            o_mut = self.game.play(mut_llvl)

            # if self.env_iden == "car_racing": bug_cond = o_org-o_mut > abs(o_org*0.05)  # reward based comparison
            if mut_type == 'rlx': bug_cond = o_org > o_mut
            else: bug_cond = o_mut > o_org # failure based comparison

            if mut_type == 'rlx' and bug_cond:
                self.guiding_init_nnstates.append(np.array(mut_llvl[0]))
                cur_guiding_states.append(mut_st)
                num_rlx_bugs += 1
            
                org_llvl = self.env.reset(org.hi_lvl_state, org.rand_state)
                o_org = self.game.play(org_llvl)
                mut_llvl = self.env.reset(mut_st, org.rand_state)
                o_mut = self.game.play(mut_llvl)
                if not o_org > o_mut: rlx_fp += 1

            elif mut_type == 'unrlx' and bug_cond:
                if org_idx not in guiding_st_idx:
                    self.guiding_init_nnstates.append(np.array(org_llvl[0]))
                    cur_guiding_states.append(org.hi_lvl_state)
                    guiding_st_idx.append(org_idx)

                    org_llvl = self.env.reset(org.hi_lvl_state, org.rand_state)
                    o_org = self.game.play(org_llvl)
                    mut_llvl = self.env.reset(mut_st, org.rand_state)
                    o_mut = self.game.play(mut_llvl)
                    if not o_mut > o_org: unrlx_fp += 1

                num_unrlx_bugs += 1
                
        if self.guiding_init_nnstates: self.all_gstates_by_test.append(self.guiding_init_nnstates)

        self.game.env.seed(self.seed)
        avg_rew = self.game.eval(eval_budget=30)

        # normal inits has to be added before test
        if self.env_iden == "car_racing":
            prev_alpha = self.env.venv.guide_prob
            if len(self.guiding_init_nnstates) == 0 or avg_rew < self.env.venv.guide_rew or self.train_type == "normal" : alpha = 0
            else: alpha = tune_alpha(self.guiding_init_nnstates, self.env.venv.normal_init_nnstates)
            self.env.venv.guide_prob = alpha
            self.env.venv.locked = False
            self.all_nstates_by_test.append(copy.copy(self.env.venv.normal_init_nnstates))
            self.env.venv.normal_init_nnstates.clear()
            if cur_guiding_states: self.env.venv.all_guiding_states.append(cur_guiding_states)
        else:
            prev_alpha = self.env.guide_prob
            if len(self.guiding_init_nnstates) == 0 or avg_rew < self.env.guide_rew or self.train_type == "normal" : alpha = 0
            else: alpha = tune_alpha(self.guiding_init_nnstates, self.env.normal_init_nnstates)
            self.env.guide_prob = alpha
            self.env.locked = False
            self.all_nstates_by_test.append(copy.copy(self.env.normal_init_nnstates))
            self.env.normal_init_nnstates.clear()
            if cur_guiding_states: self.env.all_guiding_states.append(cur_guiding_states)

        num_tot_bugs = num_rlx_bugs + num_unrlx_bugs

        info_f = open(self.log_dir + "/info.log", "a")
        data_f = open(self.log_dir + "/bug_rew.log", "a")

        data_f.write("%d,%d,%f,%f\n" % (self.num_timesteps, num_tot_bugs, avg_rew, prev_alpha))
        info_f.write("Current agent has %d(%d) + %d(%d) = %d(%d) bugs and %f reward at %d timesteps. Guide prob. was %f.\n" % (num_rlx_bugs, rlx_fp,
        num_unrlx_bugs, unrlx_fp, num_tot_bugs, rlx_fp+unrlx_fp, avg_rew, self.num_timesteps, prev_alpha))
        info_f.close()
        data_f.close()


    def post_train(self):

        if self.env_iden == "car_racing":
            self.all_nstates_by_test.append(copy.copy(self.env.venv.normal_init_nnstates))
        else:
            self.all_nstates_by_test.append(copy.copy(self.env.normal_init_nnstates))

        all_guide_inits = np.array(self.all_gstates_by_test)
        all_normal_inits = np.array(self.all_nstates_by_test[1:])  # there is a shift in collection of normal and guided states

        gmax_by_test, gmin_by_test = [], []
        for ginit in all_guide_inits:
            gmax_by_test.append(np.amax(ginit, axis=0))
            gmin_by_test.append(np.amin(ginit, axis=0))

        guide_max = np.amax(gmax_by_test, axis=0)
        guide_min = np.amin(gmin_by_test, axis=0)

        nmax_by_test, nmin_by_test = [], []
        for ninit in all_normal_inits:
            nmax_by_test.append(np.amax(ninit, axis=0))
            nmin_by_test.append(np.amin(ninit, axis=0))

        normal_max = np.amax(nmax_by_test, axis=0)
        normal_min = np.amin(nmin_by_test, axis=0)

        # normal_max = np.amax(all_normal_inits, axis=(0,1))
        # normal_min = np.amin(all_normal_inits, axis=(0,1))

        feature_max, feature_min = [], []
        for i in range(len(guide_max)):
            feature_max.append(max(guide_max[i], normal_max[i]))
            feature_min.append(min(guide_min[i], normal_min[i]))

        # Calculate all hashes
        cov_hash, num_unq_partitions = get_hashes(all_guide_inits, all_normal_inits, feature_min, feature_max)
        
        # Find coverage dist. of normal inits
        all_normal_cov_distribution = get_cov(cov_hash, all_normal_inits, num_unq_partitions, feature_min, feature_max)

        # Find coverage dist. of guiding inits
        all_guide_cov_distribution = get_cov(cov_hash, all_guide_inits, num_unq_partitions, feature_min, feature_max)
        
        guide_cov  = open(self.log_dir + "/guide_cov_dist.p", "wb")
        normal_cov = open(self.log_dir + "/normal_cov_dist.p", "wb")
        pickle.dump(all_guide_cov_distribution, guide_cov)
        pickle.dump(all_normal_cov_distribution, normal_cov)


    def explore(self):
        from lunar import Fuzzer, EnvWrapper

        game = EnvWrapper.Wrapper(self.env_iden)
        game.env = self.env
        game.model = self.policy
        game.action_space = range(self.env.action_space.n)

        rand_seed = self.seed  # separate rng created in Fuzzer
        fuzz_type = 'inc'
        fuzz_game = game
        inf_prob  = 0.2
        coverage  = 'raw'
        coverage_thold = 0.4
        fuzz_mut_bdgt  = 25

        fuzzer = Fuzzer.Fuzzer(rand_seed=rand_seed, fuzz_type=fuzz_type, fuzz_game=fuzz_game, inf_prob=inf_prob, coverage=coverage, coverage_thold=coverage_thold, mut_budget=fuzz_mut_bdgt)
       
        poolfile= open("fuzzer_pool_1h_guide_train.p", 'rb')
        pool = pickle.load(poolfile)
        fuzzer.pool = pool
        poolfile.close()

        self.fuzzer = fuzzer
        #self.fuzzer.fuzz()
        #print("Pool size: %d" % len(self.fuzzer.pool))

        #fuzzer_summary = open("fuzzer_pool_1h_guide_train.p", "wb")
        #pickle.dump(self.fuzzer.pool, fuzzer_summary)
