import os
import gfootball.env as football_env

# comando para executar o servidor para esse cliente:
# python -m gfootball.play_game --players "remote:left_players=3,port=6000;bot:right_players=1"


if __name__ == "__main__":
    # lembre-se de alterar o número de players também no comando play_game acima
    # esse número deve corresponder ao número de jogadores que seu agente foi treinado
    # para controlar - por padrão é 1
    players = 3

    player_config = {
        "left_players": players if os.environ.get("PLAYER_SIDE") == "left" else 0,
        "right_players": players if os.environ.get("PLAYER_SIDE") == "right" else 0,

        # CONFIGURE AQUI SEU AGENTE
        # utilize as mesmas configurações que você utilizou durante o treinamento
        "index": 0,
        "action_set": "default",
        "representation": "extracted",
        "stacked": False,
        "channel_dimensions": (42, 42),
        # ray specific settings
        "algorithm": "PPO",
        "policy_ids": ['policy_{}'.format(i) for i in range(players)],
        "checkpoint": "./PPO/PPO_gfootball_a659d_00000_0_2024-02-04_11-49-51/checkpoint_000078/checkpoint-78",
    }

    # cria o ambiente cliente, que se conecta ao servidor
    env = football_env.create_local_remote_environment(
        # configuração do agente e wrappers
        stacked=player_config["stacked"],
        channel_dimensions=player_config["channel_dimensions"],
        representation=player_config["representation"],
        action_set=player_config["action_set"],
        # informa ao ambiente como assinalar os jogadores
        number_of_left_players_agent_controls=player_config["left_players"],
        number_of_right_players_agent_controls=player_config["right_players"],
        # configuração do servidor
        server_ip="127.0.0.1",
        server_port=os.environ.get("SERVER_PORT", "6000"),
        # a transmissão do frame reduz drasticamente o FPS
        # só a habilite caso seu agente a utilize
        include_frame_in_obs=False,
    )

    # --- CARREGUE AQUI SEU AGENTE
    import pickle5 as pickle
    import ray
    from ray.rllib.env.base_env import BaseEnv

    ray.init(ignore_reinit_error=True)
    # Load the config from pickled.
    with open(os.path.join(os.path.dirname(player_config["checkpoint"]), "../params.pkl"), "rb") as f:
        config = pickle.load(f)

    # no need for parallelism on evaluation
    config["num_workers"] = 0
    config["num_gpus"] = 0
    # create a dummy env since it's required but we only care about the policy
    ray.tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
    config["env"] = "DummyEnv"
    # create the Trainer from config
    cls = ray.tune.registry.get_trainable_cls(player_config["algorithm"])
    agent = cls(env=config["env"], config=config)
    # load state from checkpoint
    agent.restore(player_config["checkpoint"])

    # --- DEFINA A FUNÇÃO DE SELEÇÃO DE AÇÃO
    def act(observation):
        # infere uma ação para o time controlado
        actions = []
        for i, policy_id in enumerate(player_config["policy_ids"]):
            obs = {policy_id: observation if players == 1 else observation[i]}
            action = agent.compute_actions(obs, policy_id=policy_id)
            actions.append(int(action[policy_id]))
        return actions

    # loop de interação com o ambiente remoto
    obs = env.reset()
    while True:
        actions = act(obs)
        obs, _, _, _ = env.step(actions)
