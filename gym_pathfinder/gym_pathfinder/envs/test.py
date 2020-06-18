from gym_pathfinder_env import PathFinderEnv

#from stable_baselines.common.env_checker import check_env

env = PathFinderEnv()
# Uncheck line below to check if the environment follows the interface.
# If itdoesn't follow the interface, an error will be thrown
#check_env(env, warn=True)

episodes = 6
n_steps = 200
for episode in range(episodes):
  obs = env.reset()
  for step in range(n_steps):
    print("\nStep {}".format(step + 1))
    obs, reward, done, info = env.step(env.optimal_action())
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode = 'human')
    if done:
      print("Goal reached!", "reward=", reward)
      break
