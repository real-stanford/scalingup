from utils import set_up_env

if __name__ == "__main__":
    env, _, _ = set_up_env("table_top")
    obs = env.reset(episode_id=0)
    obs.images["front"].show()
    obs.images["ur5e/wsg50/d435i/rgb"].show()
