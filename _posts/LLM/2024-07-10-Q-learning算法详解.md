---
layout: post
title: "Q-learning算法详解"
date: 2024-07-10
author: Cola Liu
categories: [LLM, RL]
---

## 马尔可夫决策过程 MDP
马尔可夫决策过程（Markov Decision Process, MDP）是一种数学框架，用于描述具有随机性的决策问题。在这种框架下，智能体在不同状态之间进行转换，并根据某种策略做出动作，以最大化累积奖励。MDP 是强化学习的理论基础，广泛用于机器人控制、游戏 AI 和优化问题等领域。

### MDP 的组成要素

<img src="/assets/imgs/ai/llm/RL/rl.png"/>

一个标准的 MDP 由五个要素组成：

1. **状态集 $ S $**：
   - MDP 的状态集 $ S $ 描述了环境的所有可能状态。在任意时刻，环境都会处于状态 $ s \in S $。状态可以是离散的，也可以是连续的。
   - 在迷宫游戏中，每个迷宫单元格都可以视为一个状态。

2. **动作集 $ A $**：
   - 动作集 $ A $ 描述了智能体在每个状态下可以选择的动作。

3. **状态转移概率 $ P(s' \| s, a) $**：
   - 状态转移概率描述了从状态 $ s $ 执行动作 $ a $ 后转移到下一个状态 $ s' $ 的概率。

4. **奖励函数 $ R(s, a) $**：
   - 奖励函数 $ R(s, a) $ 指定智能体在状态 $ s $ 下执行动作 $ a $ 所得到的即时奖励。奖励可以是正的、负的或零，取决于动作的好坏。

5. **折扣因子 $ \gamma $**：
   - 折扣因子 $ \gamma \in [0, 1] $ 决定了未来奖励的重要性。


接下来具体来看看 **Q-learning** 算法的原理及实现。


## Q-learning

**Q-learning** 是一种强化学习算法，用于在马尔可夫决策过程（MDP）中学习最优策略。它可以应用于迷宫游戏这样的环境，帮助智能体在迷宫中找到最短路径或最优路径。


### Q-learning 算法步骤

1. **初始化 Q 表**：创建一个 **Q** 表，表的每个条目 **Q(s, a)** 初始为零，其中 **s** 是状态，**a** 是动作（假设我们有一个 **3x5** 的迷宫maze）。

如下图所示，每个格子里存放的值称为Q值 —— **Q(s, a)**。

<img src="/assets/imgs/ai/llm/RL/q-learning/q-table.png"/>

2. **选择动作 Choose Action**：使用 **ε-greedy** 策略选择动作。即以概率 **ε** 选择随机动作，以概率 1-ε 选择当前 **Q** 值最高的动作。

<img src="/assets/imgs/ai/llm/RL/q-learning/epsilon-greedy.png" width="200"/>

在上面例子中，假设我们处于 **state (1, 1)** 的位置上，其中 
```
Q((1, 1), up) = 0.4
Q((1, 2), down) = 0.1
Q((1, 2), left) = 0.2
Q((1, 2), right) = 0.3
```
那么接下来选择的 **action** 应该是Q值最大的 **up**。


3. **执行动作 Take Action**：在当前状态下执行所选动作，并观察下一个状态及奖励。

4. **更新 Q 值**：
   $
   Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
   $
   其中：
   - $ \alpha $ 是学习率，控制 Q 值更新的幅度。
   - $ r $ 是奖励。
   - $ \gamma $ 是折扣因子，表示未来奖励的重要性。
   - $ \max_{a'} Q(s', a') $ 是下一个状态中所有可能动作的最大 Q 值。

5. **重复**：直到智能体收敛，即 Q 值变化很小或达到预定的迭代次数。

6. **策略提取**：从 Q 表中提取最优策略，即在每个状态下选择 Q 值最高的动作。


### 迷宫游戏中的应用
接下来我们举一个具体的例子来进一步理解 **Q-learning** 算法。

在迷宫游戏中，Q-learning 可以帮助智能体学会如何在迷宫中找到出口。通过不断地探索和更新 Q 值，智能体会逐渐找到最短路径或最优路径。

#### 1、Q表
已知一个我们有一个 3x5 的迷宫maze，建立一个Q表，用来存放Q值。以s为(1, 1)为例，Q 表和 Q(s, a) 如下图所示 ⬇️

<img src="/assets/imgs/ai/llm/RL/q-learning/q-s-a.png" width="300"/>

#### 2、Q值的更新
我们知道，Q值的更新公式如下：

 $
   Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
   $

假设我们选择的动作(choose action时)是 **up**，同样以s为 `(1, 1)` 为例，那么s'为 `(0, 1)`。 代入上述公式如下：

 $
   Q((1,1), up) \leftarrow Q((1,1), up) + \alpha \left( r + \gamma \max_{a'} Q((0, 1), a') - Q((1,1), up) \right)
   $

> $\max_{a'} Q((0, 1), a')$ 即为` Q((0, 1), up)、Q((0, 1), down)、Q((0, 1), left)、Q((0, 1), right) `四者中的最大 Q 值。

<img src="/assets/imgs/ai/llm/RL/q-learning/q-learning-up.png"/>

#### 3、整体步骤
有了上述的原理讲解，再回过头来看 **Q-learning** 算法的整体流程就变得简单多了。

首先，我们使用 **ε-greedy** 策略选择动作，这里决定 **Q(s,a)** 中的**a**。当 **a** 确定后，可以确定下一个状态，即**Q(s', a')** 的 **s'**。

接着，根据公式更新 Q 值，如下图所示。


<img src="/assets/imgs/ai/llm/RL/q-learning/q-learning-all.png"/>

> 注意不是所有的 **Q(s,a)** 都会更新，而是 **Choose Action** 步骤中选择到的动作 **action**， 当 **a == action** 的时候才会更新。

然后，把 **Q(s', a')** 赋值给 **Q(s, a)**，重复以上过程直到到达终点。

以上过程为一个完整的 **episode（回合）**。可以根据具体情况选择训练的回合数目，重复Q值更新过程即可。

#### 代码实现
```python
import numpy as np
import random

# 创建3x5迷宫环境
class MazeEnv:
    def __init__(self):
        # 定义迷宫：-1是可走路径，-100是障碍物，100是终点
        self.maze = np.array([
            [-1, -1, -1, -100, 100],
            [-1, -1, -100, -1, -1],
            [-1, -1, -1, -1, -1]
        ])
        self.start = (0, 0)  # 起点位置
        self.state = self.start
        self.actions = ['up', 'down', 'left', 'right']
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(2, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(4, y + 1)

        if self.maze[x, y] == -100:  # 遇到障碍物
            next_state = self.state  # 保持当前位置
        else:
            next_state = (x, y)

        reward = -1
        done = False
        if self.maze[x, y] == 100:  # 到达终点
            reward = 100
            done = True

        self.state = next_state
        return next_state, reward, done
    
    def is_terminal(self, state):
        x, y = state
        return self.maze[x, y] == 100

# 初始化 Q 表
Q_table = np.zeros((3, 5, 4))  # 3x5 的状态空间，4个动作

# 参数设置
learning_rate = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
episodes = 100  # 训练回合数

env = MazeEnv()

# 选择动作：ε-贪心策略
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:  # 探索
        action = np.random.choice(env.actions)
    else:  # 利用
        action_index = np.argmax(Q_table[state[0], state[1], :])
        action = env.actions[action_index]
    return action

# 训练过程
for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)
        
        action_idx = env.actions.index(action)
        next_max = np.max(Q_table[next_state[0], next_state[1], :])
        
        # Q-learning 更新
        Q_table[state[0], state[1], action_idx] += learning_rate * (reward + gamma * next_max - Q_table[state[0], state[1], action_idx])
        
        state = next_state

# 输出训练后的 Q 表
for row in range(3):
    for col in range(5):
        if env.maze[row, col] == -100:
            print("####", end="  ")  # 障碍物
        elif env.maze[row, col] == 100:
            print(" G ", end="  ")  # 终点
        else:
            action_idx = np.argmax(Q_table[row, col, :])
            action = env.actions[action_idx]
            print(f" {action[0].upper()} ", end="  ")  # 输出最优动作（U,D,L,R）
    print()

```


上述示例，通过 **Q-learning** 算法，智能体不断进行探索和利用，最终学会在迷宫中找到最优路径。ε-贪心策略确保智能体在探索新路径和利用已有知识之间保持平衡。



