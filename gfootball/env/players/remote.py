
import json

from gfootball.env import player_base, socket_util

class Player(player_base.PlayerBase, socket_util.Server):
    def __init__(self, player_config, env_config):
        # init super player class first
        player_base.PlayerBase.__init__(self, player_config)
        # init super server
        socket_util.Server.__init__(self, int(player_config.get("port", "5000")))

        self.player_config = player_config
        self.env_config = env_config

        # prepare configs
        configs = json.dumps(
            {
                "player_config": self.player_config,
                "env_config": self.env_config.get_dictionary(),
            }
        )
        # send configs
        self.sendall(configs, prefix="C")

    def take_action(self, observation):
        # Send the observation to the client
        obs = json.dumps(observation, cls=socket_util.NumpyEncoder)
        self.sendall(obs, prefix="O")

        # Receive action from the client
        action = json.loads(self.recvall(prefix="A"))
        return action

    def reset(self):
        pass  # Add any reset behavior if needed



