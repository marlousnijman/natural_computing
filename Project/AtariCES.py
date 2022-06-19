import gym
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from model import DQN_Chrabaszcz_model, DQN_Rodrigues_model, DQN_Mnih_model

class AtariCES():
    def __init__(self, game, render,
                max_step=1000, sigma=0.01,
                n_parents=10, n_offspring=20, 
                iterations=10, parent_selection="topn", 
                adaptive_type='constant'):
        """
        Initialize the Atari Canonical Evolutionary
        Strategy class.
        """

        # Game parameters
        self.game = game
        self.render = render
        self.max_step = max_step
        self.n_actions = 0

        # Evolutionary strategy parameters
        self.sigma = sigma
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.iterations = iterations
        self.parent_selection = parent_selection
        self.adaptive_type = adaptive_type
        
    def set_model(self, model):
        self.model = model

    def initiate_env(self):
        """
        Initatie the Atari environment with preprocessing
        wrappers.
        """
        env = gym.make(f'{self.game}NoFrameskip-v4')
        env = AtariPreprocessing(env, noop_max=30)
        env = FrameStack(env, num_stack=4)
        env.reset()

        n_actions = env.action_space.n
        actions_meanings = env.env.get_action_meanings()
        state_dim = env.observation_space.shape
        self.n_actions = n_actions

        return env, n_actions, actions_meanings, state_dim

    def plot_frames(self, frames):
        """
        Plot environment frames.
        """
        fig, axs = plt.subplots(1, frames.shape[2])
        for i, ax in enumerate(axs.flat):
            ax.imshow(frames[:,:,i], cmap="gray")
            ax.axis("off")
            ax.set_title(f"frame {i+1}")

    def get_frames(self, observation):
        """
        Get frames in the right format to be used as model
        input.
        """
        observation = observation.__array__().transpose(1, 2, 0)
        observation = np.expand_dims(observation, axis=0)

        return observation

    def episode(self, model, max_step, render=False):
        """
        Perform an episode of a maximum number of steps
        for the specified game and return the reward. 
        """
        if render:
            env = gym.make('{}NoFrameskip-v4'.format(self.game), render_mode='human')
        else:
            env = gym.make('{}NoFrameskip-v4'.format(self.game))

        env = AtariPreprocessing(env, noop_max=30)
        env = FrameStack(env, num_stack=4)
        frames = self.get_frames(env.reset())

        episode_reward = 0
        step = 0

        while step < self.max_step:
            if render:
                env.render(mode="rgb_array")
            step += 1



            action = np.argmax(model(frames).numpy()) 
            frames, reward, done, info = env.step(action)
            frames = self.get_frames(frames)

            episode_reward += reward

            if done:
                frames = self.get_frames(env.reset())

        return episode_reward

    def get_start_parameters(self, model):
        """
        Get the starting parameters, initalized with weights
        drawn from a normal distribution. 
        """
        parameters = model.trainable_weights
        parameters = np.concatenate(parameters, axis=None)
        start_weights = np.random.normal(0, 0.05, parameters.shape)

        return start_weights
        
    def get_weights(self, method):
        """
        Get weights used to compute the weighted mean of
        model parameters if selecting top n parents.
        Else, compute uniform weights.
        """
        if method == "topn":
            W = [np.log(self.n_parents - 0.5) - np.log(i) for i in range(1, self.n_parents + 1)]
            W /= np.sum(W)

            return W

        elif (method == "random") or (method == "tournament"): 
            W = [1 / self.n_parents] * self.n_parents
            return W

        else:
            print("Invalid parent selection method")

    def set_model_weights(self, model, theta, sigma, e):
        """
        Set the model weights based on theta and
        possibly random noise.
        """

        current_model = model
        parameters = current_model.trainable_weights
        start_idx = 0
        w = theta + sigma * e

        for p in parameters:
            n = len(p.numpy().flatten()) if len(p.shape) > 1 else len(p.numpy())
            p.assign(w[start_idx:(start_idx + n)].reshape(p.shape))
            start_idx += n

        return current_model

    def show_last(self, theta, model, max_step):
        """
        Show performance after the final generation.
        """
        new_model = self.set_model_weights(model, theta, 0, 0)
        self.episode(new_model, max_step, render=True)

        print("Finished")

    def select_parents(self, e, rewards, method):
        """ 
        Select parents according to the method of choice.
        Can be selecting the top n best parents, random parents,
        or tournament selection. 
        """
        if method == "topn":
            sorted_rewards = rewards.argsort()[::-1]
            best_offspring = e[sorted_rewards][:self.n_parents]

            return best_offspring

        elif method == "random":
            return random.choices(e, k=self.n_parents)

        elif method == "tournament": 
            weights = rewards / np.sum(rewards)
            return random.choices(e, weights=weights, k=self.n_parents)

        else:
            print("Invalid parent selection method")

    def get_sigma(self,sigma,i,iterations,adaptive_type):
        """
        Return step size according the adaptive mutation strategy.
        """
        if adaptive_type == 'constant':
            real_sigma = sigma

        elif adaptive_type == 'linear':
            real_sigma = np.linspace(sigma[0],sigma[1],iterations)[i]

        elif adaptive_type == 'exp':
            real_sigma = np.flip(np.logspace(sigma[0],sigma[1],iterations))[i]

        elif adaptive_type == 'log':
            real_sigma = np.logspace(sigma[1], sigma[0], iterations)[i]

        else:
            real_sigma = sigma
            
        return real_sigma

    def plot_rewards(self, best_r, worst_r, mean_r):
        """
        Plot the mean reward over iterations and an interval
        showing the best and worst reward.
        """
        x = np.arange(len(best_r))
        plt.figure()
        plt.plot(x, mean_r)
        plt.fill_between(x, worst_r, best_r, color="#C4E1F5")
        plt.xticks(x)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.savefig(f"rewards_{self.game}_{self.adaptive_type}.png")
        plt.show()

    def ensemble_CES(self):
        """
        Perform the Canonical Evolutionary Strategy with different DQN models using ensemble 
        learning on the predefined Atari game
        """
        chrabaszcz_model = DQN_Chrabaszcz_model(n_actions=self.n_actions) 
        rodrigues_model = DQN_Rodrigues_model(n_actions=self.n_actions) 
        mnih_model = DQN_Mnih_model(n_actions=self.n_actions) 
        
        chrabaszcz_theta = self.get_start_parameters(chrabaszcz_model)
        rodrigues_theta = self.get_start_parameters(rodrigues_model)
        mnih_theta = self.get_start_parameters(mnih_model)

        W = self.get_weights(method=self.parent_selection)

        best_r = np.zeros((self.iterations))
        worst_r = np.zeros((self.iterations))
        mean_r = np.zeros((self.iterations))

        for t in range(self.iterations):
            print('Iteration: ', t + 1)

            chrabaszcz_e = np.zeros((self.n_offspring, chrabaszcz_theta.shape[0]))
            rodrigues_e = np.zeros((self.n_offspring, rodrigues_theta.shape[0]))
            mnih_e = np.zeros((self.n_offspring, mnih_theta.shape[0]))

            r = np.zeros((self.n_offspring))
            sigma_step = self.get_sigma(self.sigma, t, self.iterations, self.adaptive_type)
            
            for i in tqdm(range(self.n_offspring)):
                chrabaszcz_e[i] = np.random.normal(0, sigma_step**2, size=chrabaszcz_theta.shape)
                rodrigues_e[i] = np.random.normal(0, sigma_step**2, size=rodrigues_theta.shape)
                mnih_e[i] = np.random.normal(0, sigma_step**2, size=mnih_theta.shape)

                new_chrabaszcz_model = self.set_model_weights(chrabaszcz_model, chrabaszcz_theta, sigma_step, chrabaszcz_e[i])
                new_rodrigues_model = self.set_model_weights(rodrigues_model, rodrigues_theta, sigma_step, rodrigues_e[i])
                new_mnih_model = self.set_model_weights(mnih_model, mnih_theta, sigma_step, mnih_e[i])

                if self.render:
                    chrabaszcz_r = self.episode(new_chrabaszcz_model, self.max_step, render=True)
                    rodrigues_r = self.episode(new_rodrigues_model, self.max_step, render=True)
                    mnih_r = self.episode(new_mnih_model, self.max_step, render=True)

                    r[i] = np.max([chrabaszcz_r, rodrigues_r, mnih_r])
                else:
                    chrabaszcz_r = self.episode(new_chrabaszcz_model, self.max_step)
                    rodrigues_r = self.episode(new_rodrigues_model, self.max_step)
                    mnih_r = self.episode(new_mnih_model, self.max_step)

                    r[i] = np.max([chrabaszcz_r, rodrigues_r, mnih_r])

            best_r[t] = np.max(r)
            worst_r[t] = np.min(r)
            mean_r[t] = np.mean(r)

            print(f"best reward: {best_r[t]}")

            chrabaszcz_best_es = self.select_parents(chrabaszcz_e, r, method=self.parent_selection)
            chrabaszcz_theta += sigma_step * np.sum([W[i] * chrabaszcz_best_es[i] for i in range(len(W))], axis=0)

            rodrigues_best_es = self.select_parents(rodrigues_e, r, method=self.parent_selection)
            rodrigues_theta += sigma_step * np.sum([W[i] * rodrigues_best_es[i] for i in range(len(W))], axis=0)

            mnih_best_es = self.select_parents(mnih_e, r, method=self.parent_selection)
            mnih_theta += sigma_step * np.sum([W[i] * mnih_best_es[i] for i in range(len(W))], axis=0)

        self.plot_rewards(best_r, worst_r, mean_r)
        
        theta_list = [chrabaszcz_theta, rodrigues_theta, mnih_theta]
        model_list = [chrabaszcz_model, rodrigues_model, mnih_model]

        return theta_list, model_list, best_r



            



            








