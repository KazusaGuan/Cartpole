import gymnasium as gym
import agent
env = gym.make('CartPole-v1', render_mode='human')
agent1 = agent.agent_for_test(env)
agent1.read_model()
observation, info = env.reset()
episode_over = False
total_reward = 0
while not episode_over:
    action = agent1.get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f'共有{agent1.unknown}次选择未知')
print(f'episode finished!{total_reward}')

