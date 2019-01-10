import gym
import sys
import autograd.numpy as np
import seaborn
import random
import pickle
import matplotlib.pyplot as plt


def sigmoid(x: np.array) -> np.array:
    return np.exp(x) / (np.exp(x) + 1)


def relu(x: np.array) -> np.array:
    return np.maximum(x, 0.8)


def M(P: [[np.array], np.array,
          np.array], obs: np.array, lower_bounds: np.array,
      upper_bounds: np.array, stochastic: bool) -> np.array:
    L, M, V = P
    """ Append constant term. """
    obs = np.append(obs, 1)
    for layer in L:
        obs = sigmoid(layer @ obs)
    """ 
    Act as if Every parameter belongs to a special exp distribution.  
    Autograd cannot handle normal distribution so exp will do.
    """

    if stochastic:
        uniform_random = np.clip(np.random.rand(len(upper_bounds)), 0.1, 0.90)
        #         -||-
        AC = lower_bounds + sigmoid(
            #      Inverse Exponential Sampling
            -np.log(1 - uniform_random) / relu(V @ obs) +
            # Shift                 -||-
            (M @ obs)) * (upper_bounds - lower_bounds)

        return AC

    else:
        return lower_bounds + sigmoid(M @ obs) * (upper_bounds - lower_bounds)

    return obs


def msq(*args, a=None):
    return np.sum(np.square(M(*args) - a))


class ltesearch():
    def __init__(self,
                 obs_space: int,
                 act_space: int,
                 lower_bounds: np.array,
                 upper_bounds: np.array,
                 population: int,
                 memory: int,
                 hidden_layer: int,
                 kernel: "f(obs, obs) -> metric",
                 err: "f(*Margs, a=a) -> float" = msq,
                 f: int = 5,
                 G: int = 3):
        """ Sanity Checks. """
        if population < 1:
            raise Exception("Cannot have population smaller than 1")
        """ Action Bounds. """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        """ Number of Agents. """
        self.population = population
        """ Look backs in Entropy Search. """
        self.memory = memory
        """Kernel should be semi-positive definite"""
        self.kernel = kernel
        """ Err Function """
        self.err = err
        self.grad = grad(err)
        """ F cap. """
        self.f = f
        """ G steps. """
        self.G = G

        self._stochastic = True
        """ Agents +1 for constant """
        self._agents = [
            {
                "P": [
                    # Hidden Layers.
                    [np.random.rand(hidden_layer, obs_space + 1) - 0.5],
                    # Mean Value Layer.
                    np.random.rand(act_space, hidden_layer) - 0.5,
                    # Variance Layer.
                    np.abs(np.random.rand(act_space, hidden_layer) + 0.5)
                ],
                "S":
                0,
                "F":
                0
            } for _ in range(population)
        ]
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
                     self.upper_bounds, self._stochastic)
        else:
            act = M(self._agents[self._active]["P"], obs, self.lower_bounds,
                    self.upper_bounds, self._stochastic)

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
        evolution = []
        for agent_id, data in enumerate(self._train_data):
            fitness = 0
            """ Exploration Score. """
            for frame_index, frame in enumerate(data["sa"]):
                for past_frame in data["sa"][max(0, frame_index - self.memory):
                                             frame_index]:
                    fitness += self.kernel(frame["obs"], past_frame["obs"])
            """ + Environment Reward & Rescale to make that matter """
            fitness = data["rew"] + min(
                np.exp(data["rew"]), 1 / np.exp(
                    np.round(np.log(abs(fitness) - np.log(abs(data["rew"])))))
            ) * fitness

            evolution.append((fitness, agent_id))
        """ Remember Best Agents... For Demo.. Or What not. """
        for fitness, agent_id in evolution:
            if fitness > self.best_agent["fitness"]:
                self.best_agent["fitness"] = fitness
                self.best_agent["P"] = [[
                    np.array(layer, copy=True)
                    for layer in self._agents[agent_id]["P"][0]
                ],
                                        np.array(
                                            self._agents[agent_id]["P"][1],
                                            copy=True),
                                        np.array(
                                            self._agents[agent_id]["P"][2],
                                            copy=True)]
        """ 
        Now for training Heuristic
        1. Everyone that has improved enough takes G gradient steps
        2. Everyone that has not improved enough but improved recently takes a 
           reversed gradient step
        3. Everyone that has not improved in a while ... Do something cool
        """

        def gradient_step(agent_id: int, S: float):
            gradients = [[
                np.zeros_like(layer)
                for layer in self._agents[agent_id]["P"][0]
            ],
                         np.zeros_like(self._agents[agent_id]["P"][1]),
                         np.zeros_like(self._agents[agent_id]["P"][2])]

            for frame in self._train_data[agent_id]["sa"]:
                """ Problem, Variance does not take part in this atm..."""
                local_gradients = self.grad(
                    self._agents[agent_id]["P"],
                    frame["obs"],
                    self.lower_bounds,
                    self.upper_bounds,
                    True,
                    a=frame["act"])

                gradients[0] = [
                    g + g_ for g, g_ in zip(gradients[0], local_gradients[0])
                ]
                gradients[1] += local_gradients[1]
                gradients[2] += local_gradients[2]

            num_grad = len(self._train_data[agent_id]["sa"])

            for layer, grad in zip(self._agents[agent_id]["P"][0],
                                   gradients[0]):
                layer += S * grad / num_grad

            self._agents[agent_id]["P"][1] += S * gradients[1] / num_grad
            self._agents[agent_id]["P"][2] += S * gradients[2] / num_grad

        for fitness, agent_id in evolution:
            if fitness > self._agents[agent_id]["S"]:
                print("Agent: {} -- Improved! S: {} - fitness: {}".format(
                    agent_id, self._agents[agent_id]["S"], fitness))

                self._agents[agent_id]["S"] = (
                    self._agents[agent_id]["S"] + fitness) / 2

                for _ in range(self.G):
                    gradient_step(agent_id, -1)

            elif fitness < self._agents[agent_id]["S"]\
                    and self._agents[agent_id]["F"] < self.f:
                print("Agent: {} -- Worse! S: {} - fitness: {}".format(
                    agent_id, self._agents[agent_id]["S"], fitness))

                self._agents[agent_id]["S"] = (
                    self._agents[agent_id]["S"] + fitness) / 2

                gradient_step(agent_id, 1)

            else:
                print("Agent: {} -- In a Slump! :( S: {} - fitness: {}".format(
                    agent_id, self._agents[agent_id]["S"], fitness))
                """ Do something Crazy Here!. """

                self._agents[agent_id]["S"] = (
                    self._agents[agent_id]["S"] + fitness) / 2

                self._agents[agent_id]["P"][0] = [
                    layer + np.random.normal(layer, 0.1)
                    for layer in self._agents[agent_id]["P"][0]
                ]
                self._agents[agent_id]["P"][1] += np.random.normal(
                    self._agents[agent_id]["P"][1], 0.1)
                self._agents[agent_id]["P"][2] += np.random.normal(
                    self._agents[agent_id]["P"][2], 0.1)
                """ Remember best for funzies. """

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
        # print(ac)
        obs, rew, done, _ = env.step(ac)
        t_rew += rew

    return rew


def main():
    env = gym.make("MountainCarContinuous-v0")
    # env = gym.make("Pendulum-v0")

    search = ltesearch(
        obs_space=len(env.observation_space.high),
        act_space=len(env.action_space.high),
        lower_bounds=env.action_space.low,
        upper_bounds=env.action_space.high,
        population=5,
        memory=40,
        hidden_layer=50,
        kernel=lambda x, y: np.mean(np.square(x - y)))

    if len(sys.argv) == 1:

        for _ in range(400):
            agent = search.get_agent(search.next_agent())
            agent.award(eval_env(env, agent, render=False))

        search.save_best()
        plt.show()
    elif sys.argv[1] == "-r":

        search.load_best()
        search._stochastic = False
        search.apex = True

        print(eval_env(env, search, render=True))

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    plt.show()
