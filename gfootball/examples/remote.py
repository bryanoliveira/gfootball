import gfootball.env as football_env

env = football_env.create_local_remote_environment(
    stacked=True,
    representation="extracted",
    server_ip="127.0.0.1",
    server_port="5000",
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
)
obs = env.reset()
while True:
    print("OBS", obs)
    obs, _, done, _ = env.step([0])
    if done:
        env.reset()