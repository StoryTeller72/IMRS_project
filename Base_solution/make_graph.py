import matplotlib
import matplotlib.pyplot as plt


episode, min_reward, max_reward, mean_reward = [], [], [], []

with open('Base_solution/runs/2024-12-02 02:34:22.897785.txt', 'r') as file:
    for line in file:
        ep,  max_r, min_r, mean_r = map(float, line.split())
        episode.append(ep)
        max_reward.append(max_r)
        min_reward.append(min_r)
        mean_reward.append(mean_r)


figure, axis = plt.subplots(3, figsize=(15, 5))
axis[0].plot(episode, max_reward)
axis[0].set_title("max_reward")


axis[1].plot(episode, min_reward)
axis[1].set_title("min_reward")

axis[2].plot(episode, mean_reward)
axis[2].set_title("mean_reward")
plt.show()
plt.savefig("2024-12-02 02:34:22.897785.png")
