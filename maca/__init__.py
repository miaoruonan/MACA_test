import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CollisionAvoidance-v0',
    entry_point='maca.envs.ca_env:CollisionAvoidanceEnv',
)


register(
    id='Formation-v0',
    entry_point='maca.envs.formatEnv:FormationEnv',
)