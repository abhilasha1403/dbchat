from dataclasses import dataclass, field
from typing import Optional

from proj.serve.core import BaseServeConfig

APP_NAME = "prompt"
SERVE_APP_NAME = "proj_serve_prompt"
SERVE_APP_NAME_HUMP = "proj_serve_Prompt"
SERVE_CONFIG_KEY_PREFIX = "projserve.prompt."
SERVE_SERVICE_COMPONENT_NAME = f"{SERVE_APP_NAME}_service"
# Database table name
SERVER_APP_TABLE_NAME = "proj_serve_prompt"


@dataclass
class ServeConfig(BaseServeConfig):
    """Parameters for the serve command"""

    # TODO: add your own parameters here
    api_keys: Optional[str] = field(
        default=None, metadata={"help": "API keys for the endpoint, if None, allow all"}
    )

    default_user: Optional[str] = field(
        default=None,
        metadata={"help": "Default user name for prompt"},
    )
    default_sys_code: Optional[str] = field(
        default=None,
        metadata={"help": "Default system code for prompt"},
    )
