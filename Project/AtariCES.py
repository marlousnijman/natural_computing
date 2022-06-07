import gym
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import DQN


class AtariCES():
    def __init__(self, game, render,
                max_step=1000, sigma=0.01,
                n_parents=10, n_offspring=20, 
                iterations=10):
        """
        Initialize the Atari Canonical Evolutionary
        Strategy class.
        """
        self.model = DQN()
    
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
        Perform an episode of a maximum number of stepss 
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
        
    def get_weights(self):
        """
        Get weights used to compute the weighted mean of
        model parameters.
        """
        W = [np.log(self.n_parents - 0.5) - np.log(i) for i in range(1, self.n_parents + 1)]
        W /= np.sum(W)

        return W

    def set_model_weights(self, theta, sigma, e):
        """
        Set the model weights based on theta and
        possibly random noise.
        """
        model = DQN(n_actions=self.n_actions) 
        parameters = model.trainable_weights
        start_idx = 0
        w = theta + sigma * e

        for p in parameters:
            n = len(p.numpy().flatten()) if len(p.shape) > 1 else len(p.numpy())
            p.assign(w[start_idx:(start_idx + n)].reshape(p.shape))
            start_idx += n

        return model

    def show_last(self, theta, max_step):
        """
        Show performance after the final generation.
        """
        model = self.set_model_weights(theta, 0, 0)
        self.episode(model, max_step, render=True)

        print("Finished")

    def CES(self):
        """
        Perform the Canonical Evolutionary Strategy on the
        predefined Atari game.
        """
        theta = self.get_start_parameters(self.model)
        W = self.get_weights()
        best_r = np.zeros((self.iterations))

        for t in range(self.iterations):
            print('Iteration: ', t + 1)
            e = np.zeros((self.n_offspring, theta.shape[0]))
            r = np.zeros((self.n_offspring))

            for i in tqdm(range(self.n_offspring)):
                e[i] = np.random.normal(0, self.sigma**2, size=theta.shape)
                new_model = self.set_model_weights(theta, self.sigma, e[i])
                if self.render:
                    r[i] = self.episode(new_model, self.max_step, render=True)
                else:
                    r[i] = self.episode(new_model, self.max_step)

            # print(r)
            best_rs = r.argsort()[::-1]
            print(f"reward: {r[best_rs]}")
            best_r[t] = np.max(r)
            print(f"best reward: {best_r[t]}")
            best_es = e[best_rs][:self.n_parents]

            theta += self.sigma * np.sum([W[i] * best_es[i] for i in range(len(W))], axis=0)

        return theta, best_r

