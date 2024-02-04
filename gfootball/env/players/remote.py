import json

from gfootball.env import player_base, socket_util, football_action_set


class Player(player_base.PlayerBase, socket_util.Server):
    def __init__(self, player_config, env_config):
        # init super player class first
        player_base.PlayerBase.__init__(self, player_config)
        # init super server
        socket_util.Server.__init__(self, int(player_config.get("port", "6000")))

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
        self.sendall(configs)

        # receive client config
        self.client_config = json.loads(self.recvall())

    def take_action(self, observation):
        if not self.client_config.get("include_frame_in_obs", False):
            for obs in observation:
                if "frame" in obs:
                    del obs["frame"]

        # Send the observation to the client
        obs = json.dumps(observation, cls=socket_util.NumpyEncoder)
        self.sendall(obs)

        # Receive action from the client
        actions = json.loads(self.recvall())

        return [
            football_action_set.action_set_dict[self.client_config["action_set"]][
                action
            ]
            for action in actions
        ]

    def reset(self):
        pass  # Add any reset behavior if needed
