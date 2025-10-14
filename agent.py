import gymnasium as gym
import numpy as np
from collections import defaultdict
import csv

class agent_for_train:
    
    def __init__(self,
                 env: gym.Env,
                 epsilon: float,
                 discount_factor: float,
                 lr: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 lr_decay: float,
                 final_lr: float
                 ):
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lr_decay = lr_decay
        self.final_lr = final_lr
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.training_error = []

    def change_observation(self, observation):
        return tuple(np.int32(10*np.array(observation)))

    def get_action(self, observation) -> int:
        #将连续的观测空间离散化，将(-4.8,4.8)范围内的数映射到(-480,480)的整数上
        obs = self.change_observation(observation)
        #epsilon-greedy strategy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self,
               observation,
               action,
               reward,
               terminated,
               next_observation):
        obs = self.change_observation(observation)
        next_obs = self.change_observation(next_observation)
        future_q_value = (not terminated)*np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (self.q_values[obs][action] +
                                      self.lr*temporal_difference)
        self.training_error.append(temporal_difference)

    def decay_lr(self):
        self.lr = max(self.final_lr,self.lr-self.lr_decay)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_model(self):
        write_dict_to_csv(self.q_values, 'q_values.csv')

    def read_model(self):
        read_dict_from_csv('q_values.csv')

class agent_for_test:
    def __init__(self,env: gym.Env):
        self.q_values = {}
        self.env = env
        self.unknown = 0

    def change_observation(self, observation):
        return tuple(np.int32(10*np.array(observation)))

    def get_action(self, observation) -> int:
        #将连续的观测空间离散化，将(-4.8,4.8)范围内的数映射到(-48,48)的整数上
        obs = self.change_observation(observation)
        try:
            return int(np.argmax(self.q_values[obs]))
        except:
            self.unknown += 1
            return self.env.action_space.sample()

    def read_model(self):
        self.q_values = read_dict_from_csv('q_values.csv')

def write_dict_to_csv(data_dict, filename):
    """
    将形如 {(1,2,3,4):[1,2], ...} 的字典写入CSV文件

    Args:
        data_dict: 字典，键为元组，值为列表
        filename: 输出的CSV文件名
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['tuple_data', 'list_data'])

        # 写入数据
        for key_tuple, value_list in data_dict.items():
            # 将元组转换为字符串，用 | 分隔
            tuple_str = '|'.join(map(str, key_tuple))
            # 将列表转换为字符串，用 | 分隔
            list_str = '|'.join(map(str, value_list))
            writer.writerow([tuple_str, list_str])


def read_dict_from_csv(filename):
    """
    从CSV文件读取并还原字典

    Args:
        filename: CSV文件名

    Returns:
        还原后的字典，格式为 {(1,2,3,4):[1,2], ...}
    """
    result_dict = {}

    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        # 跳过表头
        next(reader, None)

        for row in reader:
            if len(row) >= 2:
                # 解析元组数据
                tuple_str = row[0]
                if tuple_str:
                    tuple_data = tuple(float(x) for x in tuple_str.split('|'))
                else:
                    tuple_data = tuple()

                # 解析列表数据
                list_str = row[1]
                if list_str:
                    list_data = [float(x) for x in list_str.split('|')]
                else:
                    list_data = []

                # 添加到结果字典
                result_dict[tuple_data] = list_data

    return result_dict