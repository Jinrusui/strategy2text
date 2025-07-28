# Create an HackAtari environment with ram-based object detection and DQN-like observation
from hackatari import HackAtari
env = HackAtari(env_name="ALE/Pong-v5", mode="ram", obs_mode="dqn", render_mode="human")

# Interact with the environment
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Render the environment with object overlays
    env.render()