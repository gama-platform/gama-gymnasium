from gymnasium.envs.registration import register
from gama_gymnasium.gama_env import GamaEnv
register(
    id="gama_gymnasium_env/GamaEnv-v0",
    entry_point="gama_gymnasium.gama_env:GamaEnv",
)
