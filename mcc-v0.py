import gym
import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x: np.array) -> np.array:
    return np.exp(x) / (np.exp(x) + 1)


class ltesearch():
    def __init__(self, obs_space: int, act_space: int, lower_bounds: np.array,
                 upper_bounds: np.array, population: int, memory: int,
                 kernel: "f(obs, obs) -> metric", var: np.array):
        """ Sanity Checks. """
        if population < 1:
            raise Exception("Cannot have population smaller than 1")
        """ Environment Information. """
        self.obs_space = obs_space
        self.act_space = act_space
        """ Action Bounds. """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        """ Number of Agents. """
        self.population = population
        """ Look backs in Entropy Search. """
        self.memory = memory
        """Kernel should be semi-positive definite"""
        self.kernel = kernel
        """ Variance used in stochastic exploration. """
        self.var = var

        self._stochastic = True
        """ Agents """
        self._agents = [
            np.random.rand(act_space, obs_space) for _ in range(population)
        ]
        self._obs = [None for _ in range(population)]
        self._awards = [0 for _ in range(population)]
        """ Active Agent. """
        self._aid = None
        self._aobs = []
        self._aagent = None
        """ Plot data. """
        self.X = []
        self.Y = []

        self.trains = 0

    def get_agent(self, aid: int):
        self._aid = aid
        self._aobs = []
        self._aagent = self._agents[aid]
        return self

    def award(self, score: float):
        print("Score: {}".format(score))
        self._awards[self._aid] = score

    def __call__(self, obs: "obs") -> "action":
        mean = self.lower_bounds + sigmoid(self._aagent @ obs) * (
            self.upper_bounds - self.lower_bounds)

        if self._stochastic:
            action = np.random.normal(mean, self.var)
            self._aobs.append((obs, action))
            return action
        else:
            """ Mean === Mode in Normal Dist. """
            self._aobs.append((obs, mean))
            return mean

    def next_agent(self):
        if self._aid == None:
            return 0

        self._obs[self._aid] = self._aobs

        if self._aid + 1 < self.population:
            return self._aid + 1
        """ Update all agents and return agent 0. """
        agent_data = []
        for aid, obs in enumerate(self._obs):
            score = 0
            cumulative_gradient = 0
            """ Exploration Score. """
            for i, (aob, act) in enumerate(obs):
                for (ao, _) in obs[max(0, len(obs) - self.memory):]:
                    score += self.kernel(aob, ao)
                """ + Environment Reward. """
                score += self._awards[aid]
                """ Calculate Gradients. """
                mean = self.lower_bounds + sigmoid(self._agents[aid] @ aob) * (
                    self.upper_bounds - self.lower_bounds)
                """ Minimize This. """
                # Err = np.square(act - mean)
                """ Derivative of Mean Squared Error. """
                derr_dmean = -2 * (act - mean)

                out = self._agents[aid] @ aob
                """ Derivative of mean (Aka Scaled Sigmoid) """
                dmean_dout = sigmoid(out) * (1 - sigmoid(out)) * (
                    self.upper_bounds - self.lower_bounds)
                """ Derivative of Weights """
                dout_dw = aob
                """ Le - Chain Rule Derr/Dw = Derr/Dmean * Dmean/dout * Dout/Dw """
                gradient = derr_dmean * dmean_dout * dout_dw

                cumulative_gradient += gradient
            """ Maybe clean the gradient here using some nice stuff?. """
            avg_grad = cumulative_gradient / len(obs)

            agent_data.append((score, aid, avg_grad))

        agent_data.sort(reverse=True)

        num_apex = max(1, int(self.population * 1))

        apex = agent_data[:num_apex]
        losers = agent_data[num_apex:]
        """ Remove. """
        apex.sort(key=lambda x: x[1])

        avg_apex_score = 0
        """ All Apexes take a gradient Step """
        for score, aid, grad in apex:
            grad = grad * np.float_power(2, -self.trains / 10)
            print("Apex: {} -- Score: {} -- Agent: {} -- Grad: {}".format(
                aid, score, self._agents[aid], grad))
            self._agents[aid] -= grad
            avg_apex_score += score
        print("Average Apex Score: {}".format(avg_apex_score))
        """ Maybe do something cooler here? """
        for _, aid, _ in losers:
            self._agents[aid] += (
                np.random.rand(self.act_space, self.obs_space) - 0.5)

        self.X.append(len(self.X))
        self.Y.append(avg_apex_score)
        plt.plot(self.X, self.Y)
        plt.pause(0.001)

        self.trains += 1

        return 0


def eval_env(env: gym.Env, agent: "f(obs) -> action", render=True) -> float:
    done = False
    obs = env.reset()

    t_rew = 0
    while not done:
        if render:
            env.render()
        obs, rew, done, _ = env.step(agent(obs))
        t_rew += rew

    return rew


def main():
    env = gym.make("MountainCarContinuous-v0")

    search = ltesearch(
        obs_space=len(env.observation_space.high),
        act_space=len(env.action_space.high),
        lower_bounds=env.action_space.low,
        upper_bounds=env.action_space.high,
        population=1,
        memory=40,
        kernel=lambda x, y: np.sum(np.abs(x - y)),
        var=0.05 * env.action_space.high - env.action_space.low)

    for _ in range(1000):
        agent = search.get_agent(search.next_agent())
        agent.award(eval_env(env, agent, render=False))

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    plt.show()
