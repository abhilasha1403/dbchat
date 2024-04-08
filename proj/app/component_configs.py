from __future__ import annotations

import logging

from proj._private.config import Config
from proj.app.base import WebServerParameters
from proj.component import SystemApp
from proj.configs.model_config import MODEL_DISK_CACHE_DIR
from proj.util.executor_utils import DefaultExecutorFactory

logger = logging.getLogger(__name__)

CFG = Config()


def initialize_components(
    param: WebServerParameters,
    system_app: SystemApp,
    embedding_model_name: str,
    embedding_model_path: str,
):
    # Lazy import to avoid high time cost
    from proj.app.initialization.embedding_component import _initialize_embedding_model
    from proj.app.initialization.serve_initialization import register_serve_apps
    from proj.model.cluster.controller.controller import controller

    # Register global default executor factory first
    system_app.register(
        DefaultExecutorFactory, max_workers=param.default_thread_pool_size
    )
    system_app.register_instance(controller)

    # system_app.register_instance(multi_agents)

    _initialize_embedding_model(
        param, system_app, embedding_model_name, embedding_model_path
    )
    _initialize_model_cache(system_app)
    _initialize_awel(system_app, param)
    # Register serve apps
    register_serve_apps(system_app, CFG)


def _initialize_model_cache(system_app: SystemApp):
    from proj.storage.cache import initialize_cache

    if not CFG.MODEL_CACHE_ENABLE:
        logger.info("Model cache is not enable")
        return

    storage_type = CFG.MODEL_CACHE_STORAGE_TYPE or "disk"
    max_memory_mb = CFG.MODEL_CACHE_MAX_MEMORY_MB or 256
    persist_dir = CFG.MODEL_CACHE_STORAGE_DISK_DIR or MODEL_DISK_CACHE_DIR
    initialize_cache(system_app, storage_type, max_memory_mb, persist_dir)


def _initialize_awel(system_app: SystemApp, param: WebServerParameters):
    from proj.configs.model_config import _DAG_DEFINITION_DIR
    from proj.core.awel import initialize_awel

    # Add default dag definition dir
    dag_dirs = [_DAG_DEFINITION_DIR]
    if param.awel_dirs:
        dag_dirs += param.awel_dirs.strip().split(",")
    dag_dirs = [x.strip() for x in dag_dirs]

    initialize_awel(system_app, dag_dirs)