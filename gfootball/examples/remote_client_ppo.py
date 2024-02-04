from absl import app
from absl import flags

import gfootball.env as football_env

from gfootball.env.players.ppo2_cnn import Player as PPO2Player

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 6000, "Port to connect to GFootball server.")
flags.DEFINE_enum(
    "side", "left", ["left", "right"], "Side of the field the player wil play"
)

def main(_):
    # TODO: check >1 player control support
    players = 1
    player_config = {
        "index": 0,
        "action_set": "default",
        "stacked": True,
        "policy": "gfootball_impala_cnn",
        "checkpoint": "academy_run_to_score_with_keeper_v2",
        "left_players": players if FLAGS.side == "left" else 0,
        "right_players": players if FLAGS.side == "right" else 0,
    }

    env = football_env.create_local_remote_environment(
        stacked=player_config["stacked"],
        representation="extracted",
        action_set=player_config["action_set"],
        server_ip="127.0.0.1",
        server_port=FLAGS.port,
        number_of_left_players_agent_controls=player_config["left_players"],
        number_of_right_players_agent_controls=player_config["right_players"],
        include_frame_in_obs=False,
    )
    agent = PPO2Player(player_config, env.env_config)

    obs = env.reset()
    print(len(obs))
    while True:
        action = agent._policy.step(obs, deterministic=True)[0][0]
        actions = [int(action)]

        obs, _, done, _ = env.step(actions)
        if done:
            env.reset()

if __name__ == '__main__':
  app.run(main)