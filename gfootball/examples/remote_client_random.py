import gfootball.env as football_env

env = football_env.create_local_remote_environment(
    stacked=True,
    representation="extracted",
    server_ip="127.0.0.1",
    server_port="6000",
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
    include_frame_in_obs=False,
)
obs = env.reset()
while True:
    obs, _, done, _ = env.step([env.action_space.sample()])
    if done:
        env.reset()
