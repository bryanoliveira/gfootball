from absl import app
from absl import flags

import gfootball.env as football_env

from gfootball.env.players.ppo2_cnn import Player as PPO2Player

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 6000, "Port to connect to GFootball server.")
flags.DEFINE_enum(
    "side", "left", ["left", "right"], "Side of the field the player wil play"
)

# comando para executar o servidor para esse cliente:
# python -m gfootball.play_game --players "remote:left_players=1,port=6000;bot:right_players=1"


def main(_):
    # lembre-se de alterar o número de players também no comando play_game acima
    players = 1

    player_config = {
        "left_players": players if FLAGS.side == "left" else 0,
        "right_players": players if FLAGS.side == "right" else 0,
        # CONFIGURE AQUI SEU AGENTE
        # utilize as mesmas configurações que você utilizou durante o treinamento
        "index": 0,
        "action_set": "default",
        "representation": "extracted",
        "stacked": True,
        "policy": "gfootball_impala_cnn",
        "checkpoint": "academy_run_to_score_with_keeper_v2",
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
        server_port=FLAGS.port,
        # a transmissão do frame reduz drasticamente o FPS
        # só a habilite caso seu agente a utilize
        include_frame_in_obs=False,
    )

    # --- CARREGUE AQUI SEU AGENTE
    agent = PPO2Player(player_config, env.env_config)

    # --- DEFINA A FUNÇÃO DE SELEÇÃO DE AÇÃO
    def act(obs):
        # infere uma ação para o jogador controlado
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
        obs, _, done, _ = env.step(actions)
        if done:
            env.reset()


if __name__ == "__main__":
    app.run(main)
