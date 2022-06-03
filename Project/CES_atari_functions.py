import gym
import numpy as np
from model import DQN
from gym.wrappers import AtariPreprocessing, FrameStack
import matplotlib.pyplot as plt
from tqdm import tqdm

class CES_atari_functions():
    def __init__(self,game,render):
        self.game = game
        self.render = render

    def initiate_env(self):
        env = gym.make('{}NoFrameskip-v4'.format(self.game))
        env = AtariPreprocessing(env, noop_max=30)
        env = FrameStack(env, num_stack=4)
        env.reset()
        n_actions = env.action_space.n
        actions_meanings = env.env.get_action_meanings()
        state_dim = env.observation_space.shape
        return env, n_actions, actions_meanings, state_dim

    # def env_details():
    #     print(f"Number of actions: {n_actions}")
    #     print(f"Action meanings: {actions_meanings}")
    #     print(f"State dimensions: {state_dim}")

    def plot_frames(self,frames):
        fig, axs = plt.subplots(1, frames.shape[2])
        for i, ax in enumerate(axs.flat):
            ax.imshow(frames[:,:,i], cmap="gray")
            ax.axis("off")
            ax.set_title(f"frame {i+1}")
        plt.savefig("processed_input.png")

    def get_frames(self,observation):
        observation = observation.__array__().transpose(1, 2, 0)
        observation = np.expand_dims(observation, axis=0)

        return observation

    def episode(self,model, max_step=1000,render=False):
        if render:
            env = gym.make('{}NoFrameskip-v4'.format(self.game), render_mode='human')
        else:
            env = gym.make('{}NoFrameskip-v4'.format(self.game))

        env = AtariPreprocessing(env, noop_max=30)
        env = FrameStack(env, num_stack=4)
        frames = self.get_frames(env.reset())

        episode_reward = 0
        step = 0

        while step < max_step:
            if render:
                env.render(mode="rgb_array")
            step += 1

            action = np.argmax(model(frames).numpy())
            # print(model(frames).numpy())
            # print(action)
            frames, reward, done, info = env.step(action)
            frames = self.get_frames(frames)

            episode_reward += reward
            if done:
                frames = self.get_frames(env.reset())

        return episode_reward


    def get_weights(self,parents):
        W = [np.log(parents - 0.5) - np.log(i) for i in range(1, parents + 1)]
        W /= np.sum(W)

        return W

    def get_start_parameters(self,model):
        parameters = model.trainable_weights
        parameters = np.concatenate(parameters, axis=None)
        start_weights = np.random.normal(0, 0.05, parameters.shape)

        return start_weights


    def get_model_weights(self,theta, mut_stepsize, e):
        model = DQN() ##??
        parameters = model.trainable_weights
        start_idx = 0
        w = theta + mut_stepsize * e

        for p in parameters:
            n = len(p.numpy().flatten()) if len(p.shape) > 1 else len(p.numpy())
            p.assign(w[start_idx:(start_idx + n)].reshape(p.shape))
            start_idx += n

        return model

    def show_last(self, theta, max_step=2000):

        model = self.get_model_weights(theta,0,0)
        self.episode(model,max_step,render=True)

        return "Finished"

    def CES(self, model, mut_stepsize, parents, n_offspring, iterations):

        theta = self.get_start_parameters(model)
        W = self.get_weights(parents)
        best_r = np.zeros((iterations))
        print(theta[:10])

        for t in range(iterations):
            print('Iteration: ', t + 1)
            e = np.zeros((n_offspring, theta.shape[0]))
            r = np.zeros((n_offspring))

            for i in tqdm(range(n_offspring)):
                e[i] = np.random.normal(0, 1, size=theta.shape)
                new_model = self.get_model_weights(theta, mut_stepsize, e[i])
                if self.render:
                    r[i] = self.episode(new_model,render=True)
                else:
                    r[i] = self.episode(new_model)

            best_rs = r.argsort()
            best_r[t] = np.max(r)
            print(f"best reward: {best_r[t]}")
            best_es = e[best_rs][:parents]

            theta += mut_stepsize * np.sum([W[i] * best_es[i] for i in range(len(W))], axis=0)
            print(theta[:10])
        return theta, best_r

