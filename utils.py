import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def evaluate_agents(num_runs, num_steps, agents, mab, agent_names=None):
    """
    Function to call each agent on a given MAB for a pre-specified number
    of runs and timesteps for each run.

    By comparing these multiple times against the same MAB, we can get a
    sense of how they would perform on average to a similar type of problem.

    The rewards at each time step are averaged across each run per agent,
    and plotted alongside each other.

    Additionally, a metric we like to care about is if the agent is able to
    identify the optimal arm, and how often it chooses it. The pct of the time
    the agent chose the optimal arm at each timestep is also plotted for
    comparison.

    Args:
        num_runs : int, number of runs to do for each agent
        num_steps : int, number of steps to walk through in each run
        agents : list of ActionValueMethod, list of instantiated agents to compare
        mab : MAB, instantiated MAB to compare agents against
        agent_names : list of str, optional list of names to add to legend of
            comparison plots, otherwise defaults to agent_00x in order of agents
    """
    if agent_names is None:
        agent_names = ['agent_{:3d}'.format(i) for i in range(len(agents))]

    steps = [i for i in range(num_steps)]

    avg_rewards = []
    avg_pct_optimals = []

    for agent in agents:
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
    Runs a UCB agent for a number of steps against an MAB and creates
    an output gif showing the estimated value of each arm according
    to the agent along with the uncertainty.

    Args:
        steps : int, number of steps to create the gif for
        ucb : UpperConfidenceBound
        mab : MAB
        outfile : str, name of output file, should end in .gif, but
            the commented out line would allow for an output of a
            .mp4 file as well
        fps : int, frames per second, divide steps by fps to get
            length in seconds of output file
    """
    num_arms = len(mab.bandits)

    arm_pulls = [0] * num_arms

    for _ in range(num_arms):
        arm = ucb.act()

        reward = mab.pull_bandit(arm)
        ucb.update(arm, reward)

    indices = np.arange(num_arms)

    ylim = max([mu for mu, sigma in mab.bandits]) * 1.7

    def draw_ucb_chart(frame_num):
        ax.clear()
        ax.set_ylim(-1, ylim)
        p1 = ax.bar(indices, ucb.get_q_values())
        p2 = ax.bar(indices, ucb.get_uncertainty_values(), bottom=ucb.get_q_values())
        ax.set_xlabel('UCB Arm')
        plt.title('UCB Average Rewards and Upper Bounds Step {:3d}'.format(frame_num))
        plt.legend((p1[0], p2[0]), ('Average Rewards', 'Uncertainty'), loc='upper right')


        arm = ucb.act()
        arm_pulls[arm] += 1
        reward = mab.pull_bandit(arm)
        ucb.update(arm, reward)

        count_text = ''
        for arm, count in enumerate(arm_pulls):
            count_text += 'Arm {}: {:03d}\n'.format(arm, count)

        ax.text(0.9, 0.4, count_text, fontsize=14, transform=plt.gcf().transFigure)

        plt.subplots_adjust(right=0.85)

    fig, ax = plt.subplots(figsize=(12, 8))
    animator = animation.FuncAnimation(fig, draw_ucb_chart, frames=range(steps))
    animator.save(outfile, dpi=80, writer='imagemagick')
    # animator.save(outfile, fps=fps, extra_args=['-vcodec', 'libx264'])
