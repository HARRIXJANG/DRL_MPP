from gym.envs.registration import register

register(
    id="Enviroments/WorldofPPv3_1",
    entry_point="Enviroments.envs:WorldofPPv3_1",
)

register(
    id="Enviroments/WorldofPPv3_1_predict",
    entry_point="Enviroments.envs:WorldofPPv3_1_predict",
)

register(
    id="Enviroments/WorldofPPv3_6_only_rough",
    entry_point="Enviroments.envs:WorldofPPv3_6_only_rough",
)
register(
    id="Enviroments/WorldofPPv3_7_without_rough",
    entry_point="Enviroments.envs:WorldofPPv3_7_without_rough",
)

register(
    id="Enviroments/WorldofPPv3_6_predict",
    entry_point="Enviroments.envs:WorldofPPv3_6_predict",
)
register(
    id="Enviroments/WorldofPPv3_7_predict",
    entry_point="Enviroments.envs:WorldofPPv3_7_predict",
)
