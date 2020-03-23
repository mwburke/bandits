import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def evaluate_agents(num_runs, num_steps, agents, mab, agent_names=None):
    """
    """
    if agent_names is None:
        agent_names = ['agent_{:3d}'.format(i) for i in range(len(agents))]

    steps = [i for i in range(num_steps)]

    avg_rewards = []
    avg_pct_optimals = []

    for name, agent in zip(agent_names, agents):
        avg_reward, avg_pct_optimal = evaluate_agent(num_runs, num_steps, agent, mab)
        avg_rewards.append(avg_reward)
        avg_pct_optimals.append(avg_pct_optimal)

    for reward in avg_rewards:
        plt.plot(steps, reward)

    plt.title('Average rewards')
    plt.legend(agent_names)
    plt.show()

    for avg_pct_optimal in avg_pct_optimals:
        plt.plot(steps, avg_pct_optimal)

    plt.title('Average pct optimal arm')
    plt.legend(agent_names)
    plt.show()


def evaluate_agent(num_runs, num_steps, agent, mab):
    rewards = np.zeros(shape=(num_runs, num_steps), dtype=np.float32)
    optimal_arm = int(np.argmax([mu for mu, sigma in mab.bandits]))
    pct_optimal = np.zeros(shape=(num_runs, num_steps), dtype=np.float32)

    for i in range(num_runs):
        agent.reset()

        for j in range(num_steps):
            action = agent.act()
            reward = mab.pull_bandit(action)
            rewards[i, j] = reward
            pct_optimal[i, j] = agent.counts[optimal_arm] / np.sum(agent.counts)
            agent.update(action, reward)

    return np.mean(rewards, axis=0), np.mean(pct_optimal, axis=0)


def ucb_race_chart(steps, ucb, mab, outfile, fps=30):
    """

    """
    num_arms = len(mab.bandits)

    for _ in range(num_arms):
        arm = ucb.act()
        reward = mab.pull_bandit(arm)
        ucb.update(arm, reward)

    indices = np.arange(num_arms)

    def draw_ucb_chart(frame_num):
        ax.clear()
        p1 = plt.bar(indices, ucb.get_q_values())
        p2 = plt.bar(indices, ucb.get_uncertainty_values(), bottom=ucb.get_q_values())
        plt.xlabel('UCB Arm')
        plt.title('UCB Average Rewards and Upper Bounds Step {:3d}'.format(frame_num))
        plt.legend((p1[0], p2[0]), ('Average Rewards', 'Uncertainty'))

        arm = ucb.act()
        reward = mab.pull_bandit(arm)
        ucb.update(arm, reward)

    fig, ax = plt.subplots(figsize=(10, 8))
    animator = animation.FuncAnimation(fig, draw_ucb_chart, frames=range(steps))
    animator.save(outfile, fps=fps, extra_args=['-vcodec', 'libx264'])
