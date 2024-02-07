import os
import gfootball.env as football_env

# comando para executar o servidor para esse cliente:
# python -m gfootball.play_game --players "remote:left_players=1,port=6000;bot:right_players=1"


if __name__ == "__main__":
    # lembre-se de alterar o número de players também no comando play_game acima
    # esse número deve corresponder ao número de jogadores que seu agente foi treinado
    # para controlar - por padrão é 1
    players = 1

    player_config = {
        "left_players": players if os.environ.get("PLAYER_SIDE") == "left" else 0,
        "right_players": players if os.environ.get("PLAYER_SIDE") == "right" else 0,

        # CONFIGURE AQUI SEU AGENTE
        # utilize as mesmas configurações que você utilizou durante o treinamento
        "index": 0,
        "action_set": "default",
        "representation": "extracted",
        "stacked": True,
        # baselines specific settings
        "policy": "gfootball_impala_cnn",
        "checkpoint": "11_vs_11_easy_stochastic_v2",
    }

    # cria o ambiente cliente, que se conecta ao servidor
    env = football_env.create_local_remote_environment(
        # configuração do agente e wrappers
        stacked=player_config["stacked"],
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
    from gfootball.env.players.ppo2_cnn import Player as PPO2Player
    agent = PPO2Player(player_config, env.env_config)

    # --- DEFINA A FUNÇÃO DE SELEÇÃO DE AÇÃO
    def act(obs):
        # infere uma ação para o time controlado
        # note que esse agente foi treinado para controlar um único jogador por vez
        if players == 1:
            return [int(agent._policy.step(obs, deterministic=True)[0][0])]
        else:
            # mas é simples fazer com que ele controle até o time todo:
            return [
                int(agent._policy.step(o, deterministic=True)[0][0]) 
                for o in obs
            ]

    # loop de interação com o ambiente remoto
    obs = env.reset()
    while True:
        actions = act(obs)
        obs, _, _, _ = env.step(actions)
