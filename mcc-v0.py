import gym
import sys
import numpy as np
import seaborn
import random
import pickle
import matplotlib.pyplot as plt


def M(P: [[np.array]], obs: np.array, lower_bounds: np.array,
      upper_bounds: np.array) -> np.array:
    L = P
    """ Append constant term. """
    obs = np.append(obs, 1)
    for layer in L:
        obs = layer @ obs

    return lower_bounds + obs * (upper_bounds - lower_bounds)


class E():
    def __init__(self, obs_space: int, act_space: int, lower_bounds: np.array,
                 upper_bounds: np.array, population: int):
        """ Sanity Checks. """
        if population < 1:
            raise Exception("Cannot have population smaller than 1")
        """ Action Bounds. """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        """ Number of Agents. """
        self.population = population
        """ Agents +1 for constant """
        self._agents = [{
            "P": [(np.random.rand(act_space, obs_space + 1) - 0.5) * 3],
        } for _ in range(population)]
        self._train_data = [{"sa": [], "rew": 0} for _ in range(population)]
        """ Active Agent. """
        self._active = None
        """ Plot data. """
        self.X = []
        self.Y = []

        self.best_agent = {"fitness": -10000000, "P": []}
        self.apex = False

    def get_agent(self, active: int) -> "ltesearch":
        self._active = active
        self._train_data[active] = {"sa": [], "rew": None}
        return self

    def award(self, rew: float):
        self._train_data[self._active]["rew"] = rew

    def __call__(self, obs: "obs") -> "action":
        if self.apex:
            return M(self.best_agent["P"], obs, self.lower_bounds,
                     self.upper_bounds)
        else:
            act = M(self._agents[self._active]["P"], obs, self.lower_bounds,
                    self.upper_bounds)

            self._train_data[self._active]["sa"].append({
                "obs": obs,
                "act": act
            })
            return act

    def next_agent(self) -> int:
        if self._active == None:
            return 0

        if self._active + 1 < self.population:
            return self._active + 1
        """ Update all agents and return agent 0. """
        for index, agent in enumerate(self._agents):
            if self._train_data[index]["rew"] > self.best_agent["fitness"]:
                self.best_agent["fitness"] = self._train_data[index]["rew"]
                self.best_agent["P"] = [
                    np.array(layer, copy=True)
                    for layer in self._agents[index]["P"]
                ]

        for agent in self._agents:
            for layer in agent["P"]:
                layer += (np.random.rand(*layer.shape) - 0.5)

        self.X.append(len(self.X))
        self.Y.append(self.best_agent["fitness"])

        seaborn.lineplot(self.X, self.Y)
        plt.pause(0.01)

        return 0

    def save_best(self):
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(self.best_agent, f)

    def load_best(self):
        with open('best_model.pkl', 'rb') as f:
            self.best_agent = pickle.load(f)
            print("BEST_AGENT", self.best_agent)


def eval_env(env: gym.Env, agent: "f(obs) -> action", render=True) -> float:
    done = False
    obs = env.reset()

    t_rew = 0
    while not done:
        if render:
            env.render()
        ac = agent(obs)
        obs, rew, done, _ = env.step(ac)
        t_rew += rew

    return rew


def main():
    env = gym.make("MountainCarContinuous-v0")
    # env = gym.make("Pendulum-v0")

    search = E(
        obs_space=len(env.observation_space.high),
        act_space=len(env.action_space.high),
        lower_bounds=env.action_space.low,
        upper_bounds=env.action_space.high,
        population=100)

    if len(sys.argv) == 1:

        for _ in range(400):
            agent = search.get_agent(search.next_agent())
            agent.award(eval_env(env, agent, render=False))

        search.save_best()
        plt.show()
    elif sys.argv[1] == "-r":

        search.load_best()
        search.apex = True

        print(eval_env(env, search, render=True))

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    plt.show()
