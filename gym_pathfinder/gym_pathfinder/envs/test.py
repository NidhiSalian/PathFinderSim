from foo_env import FooEnv

#from stable_baselines.common.env_checker import check_env

env = FooEnv()
# If the environment doesn't follow the interface, an error will be thrown
#check_env(env, warn=True)




#print(env.observation_space)
#print(env.action_space)
#print(env.action_space.sample())

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