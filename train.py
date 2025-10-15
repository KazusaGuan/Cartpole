from agent import agent_for_train as InvertedPendulumKAgent
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

# Training hyperparameters
learning_rate = 0.1
final_lr = 0.1
n_episodes = 1000000
start_epsilon = 1
epsilon_decay = start_epsilon / (n_episodes/2)
lr_decay = (learning_rate-final_lr)/(n_episodes/2)
final_epsilon = 0.1
discount_factor = 0.95

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def gauss(x, mu, sigma):
    return np.exp((-1)*((x-mu)**2/(2*sigma**2))) / np.sqrt(2*np.pi*sigma**2)

def get_reward(observation):
    return 3*gauss(observation[0],0,1.6)

def get_agent():
    env = gym.make('CartPole-v1')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = InvertedPendulumKAgent(env,
                                   discount_factor=discount_factor,
                                   epsilon=start_epsilon,
                                   epsilon_decay=epsilon_decay,
                                   final_epsilon=final_epsilon,
                                   lr=learning_rate,
                                   lr_decay=lr_decay,
                                   final_lr=final_lr)

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward += get_reward(obs)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs
        agent.decay_epsilon()
        agent.decay_lr()

    agent.save_model()

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


get_agent()