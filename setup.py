from typing import List, Tuple, Optional, Callable
import setuptools
import platform
import subprocess
import os
from enum import Enum
import urllib.request
from urllib.parse import urlparse, quote
import re
import shutil
from setuptools import find_packages
import functools

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

IS_DEV_MODE = os.getenv("IS_DEV_MODE", "true").lower() == "true"


BUILD_NO_CACHE = os.getenv("BUILD_NO_CACHE", "true").lower() == "true"

BUILD_FROM_SOURCE = os.getenv("BUILD_FROM_SOURCE", "false").lower() == "true"

BUILD_VERSION_OPENAI = os.getenv("BUILD_VERSION_OPENAI")


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [
            require.strip()
            for require in f
            if require.strip() and not require.startswith("#")
        ]


def get_latest_version(package_name: str, index_url: str, default_version: str):
    python_command = shutil.which("python")
    if not python_command:
        python_command = shutil.which("python3")
        if not python_command:
            print("Python command not found.")
            return default_version

    command = [
        python_command,
        "-m",
        "pip",
        "index",
        "versions",
        package_name,
        "--index-url",
        index_url,
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error executing command.")
        print(result.stderr.decode())
        return default_version

    output = result.stdout.decode()
    lines = output.split("\n")
    for line in lines:
        if "Available versions:" in line:
            available_versions = line.split(":")[1].strip()
            latest_version = available_versions.split(",")[0].strip()
            return latest_version

    return default_version


def encode_url(package_url: str) -> str:
    parsed_url = urlparse(package_url)
    encoded_path = quote(parsed_url.path)
    safe_url = parsed_url._replace(path=encoded_path).geturl()
    return safe_url, parsed_url.path


def cache_package(package_url: str, package_name: str, is_windows: bool = False):
    safe_url, parsed_url = encode_url(package_url)
    if BUILD_NO_CACHE:
        return safe_url

    from pip._internal.utils.appdirs import user_cache_dir

    filename = os.path.basename(parsed_url)
    cache_dir = os.path.join(user_cache_dir("pip"), "http", "wheels", package_name)
    os.makedirs(cache_dir, exist_ok=True)

    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        temp_path = local_path + ".tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            print(f"Download {safe_url} to {local_path}")
            urllib.request.urlretrieve(safe_url, temp_path)
            shutil.move(temp_path, local_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return f"file:///{local_path}" if is_windows else f"file://{local_path}"


class SetupSpec:
    def __init__(self) -> None:
        self.extras: dict = {}
        self.install_requires: List[str] = []


setup_spec = SetupSpec()


class AVXType(Enum):
    BASIC = "basic"
    AVX = "AVX"
    AVX2 = "AVX2"
    AVX512 = "AVX512"

    @staticmethod
    def of_type(avx: str):
        for item in AVXType:
            if item._value_ == avx:
                return item
        return None


class OSType(Enum):
    WINDOWS = "win"
    LINUX = "linux"


@functools.cache
def get_cpu_avx_support() -> Tuple[OSType, AVXType]:
    system = platform.system()
    cpu_avx = AVXType.BASIC
    env_cpu_avx = AVXType.of_type(os.getenv("proj_LLAMA_CPP_AVX"))

    if "windows" in system.lower():
        os_type = OSType.WINDOWS
        output = "avx2"
        print("Current platform is windows, use avx2 as default cpu architecture")
    elif system == "Linux":
        os_type = OSType.LINUX
        result = subprocess.run(
            ["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout.decode()

    if "avx512" in output.lower():
        cpu_avx = AVXType.AVX512
    elif "avx2" in output.lower():
        cpu_avx = AVXType.AVX2
    elif "avx " in output.lower():
        # cpu_avx =  AVXType.AVX
        pass
    return os_type, env_cpu_avx if env_cpu_avx else cpu_avx


def _build_wheels(
    pkg_name: str,
    pkg_version: str,
    base_url: str = None,
    base_url_func: Callable[[str, str, str], str] = None,
    pkg_file_func: Callable[[str, str, str, str, OSType], str] = None,
    supported_cuda_versions: List[str] = ["11.7", "11.8"],
) -> Optional[str]:
    """
    Build the URL for the package wheel file based on the package name, version, and CUDA version.
    Args:
        pkg_name (str): The name of the package.
        pkg_version (str): The version of the package.
        base_url (str): The base URL for downloading the package.
        base_url_func (Callable): A function to generate the base URL.
        pkg_file_func (Callable): build package file function.
            function params: pkg_name, pkg_version, cuda_version, py_version, OSType
        supported_cuda_versions (List[str]): The list of supported CUDA versions.
    Returns:
        Optional[str]: The URL for the package wheel file.
    """
    os_type, _ = get_cpu_avx_support()
    py_version = platform.python_version()
    py_version = "cp" + "".join(py_version.split(".")[0:2])
    if cuda_version not in supported_cuda_versions:
        print(
            f"Warnning: {pkg_name} supported cuda version: {supported_cuda_versions}, replace to {supported_cuda_versions[-1]}"
        )
        cuda_version = supported_cuda_versions[-1]

    cuda_version = "cu" + cuda_version.replace(".", "")
    os_pkg_name = "linux_x86_64" if os_type == OSType.LINUX else "win_amd64"
    if base_url_func:
        base_url = base_url_func(pkg_version, cuda_version, py_version)
        if base_url and base_url.endswith("/"):
            base_url = base_url[:-1]
    if pkg_file_func:
        full_pkg_file = pkg_file_func(
            pkg_name, pkg_version, cuda_version, py_version, os_type
        )
    else:
        full_pkg_file = f"{pkg_name}-{pkg_version}+{cuda_version}-{py_version}-{py_version}-{os_pkg_name}.whl"
    if not base_url:
        return full_pkg_file
    else:
        return f"{base_url}/{full_pkg_file}"


def torch_requires(
    torch_version: str = "2.0.1",
    torchvision_version: str = "0.15.2",
    torchaudio_version: str = "2.0.2",
):
    torch_pkgs = [
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
        f"torchaudio=={torchaudio_version}",
    ]
    torch_cuda_pkgs = []
    os_type, _ = get_cpu_avx_support()
   
    setup_spec.extras["torch"] = torch_pkgs
    setup_spec.extras["torch_cpu"] = torch_pkgs
    setup_spec.extras["torch_cuda"] = torch_cuda_pkgs


def core_requires():
    """
    pip install proj or pip install "proj[core]"
    """
    setup_spec.extras["core"] = [
        "aiohttp==3.8.4",
        "chardet==5.1.0",
        "importlib-resources==5.12.0",
        "python-dotenv==1.0.0",
        "cachetools",
        "pydantic<2,>=1",
    ]
    # Simple command line dependencies
    setup_spec.extras["cli"] = setup_spec.extras["core"] + [
        "prettytable",
        "click",
        "psutil==5.9.4",
        "colorama==0.4.6",
    ]
    # Just use by DB-GPT internal, we should find the smallest dependency set for run
    # we core unit test.
    # The dependency "framework" is too large for now.
    setup_spec.extras["simple_framework"] = setup_spec.extras["cli"] + [
        "pydantic<2,>=1",
        "httpx",
        "jinja2",
        "fastapi==0.98.0",
        "uvicorn",
        "shortuuid",
        # change from fixed version 2.0.22 to variable version, because other
        # dependencies are >=1.4, such as pydoris is <2
        "SQLAlchemy>=1.4,<3",
        # for cache
        "msgpack",
        # for cache
        # TODO: pympler has not been updated for a long time and needs to
        #  find a new toolkit.
        "pympler",
        "sqlparse==0.4.4",
        "duckdb==0.8.1",
        "duckdb-engine",
    ]
    # # TODO: remove fschat from simple_framework
    # if BUILD_FROM_SOURCE:
    #     setup_spec.extras["simple_framework"].append(
    #         f"fschat @ {BUILD_FROM_SOURCE_URL_FAST_CHAT}"
    #     )
    # else:
    #     setup_spec.extras["simple_framework"].append("fschat")

    setup_spec.extras["framework"] = setup_spec.extras["simple_framework"] + [
        "coloredlogs",
        "seaborn",
        # https://github.com/eosphoros-ai/DB-GPT/issues/551
        "pandas==2.0.3",
        "auto-gpt-plugin-template",
        "gTTS==2.3.1",
        "langchain>=0.0.286",
        "pymysql",
        "jsonschema",
        # TODO move transformers to default
        # "transformers>=4.31.0",
        "transformers>=4.34.0",
        "alembic==1.12.0",
        # for excel
        "openpyxl==3.1.2",
        "chardet==5.1.0",
        "xlrd==2.0.1",
        "aiofiles",
        # for agent
        "GitPython",
        # For AWEL dag visualization, graphviz is a small package, also we can move it to default.
        "graphviz",
    ]


def knowledge_requires():
    """
    pip install "proj[knowledge]"
    """
    setup_spec.extras["knowledge"] = [
        "spacy==3.5.3",
        "chromadb==0.4.10",
        "markdown",
        "bs4",
        "python-pptx",
        "python-docx",
        "pypdf",
        "python-multipart",
        "sentence-transformers",
    ]




def quantization_requires():
    pkgs = []
    os_type, _ = get_cpu_avx_support()
    if os_type != OSType.WINDOWS:
        pkgs = ["bitsandbytes"]
    else:
        latest_version = get_latest_version(
            "bitsandbytes",
            "https://jllllll.github.io/bitsandbytes-windows-webui",
            "0.41.1",
        )
        extra_index_url = f"https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-{latest_version}-py3-none-win_amd64.whl"
        local_pkg = cache_package(
            extra_index_url, "bitsandbytes", os_type == OSType.WINDOWS
        )
        pkgs = [f"bitsandbytes @ {local_pkg}"]
        print(pkgs)
    # For chatglm2-6b-int4
    pkgs += ["cpm_kernels"]
    setup_spec.extras["quantization"] = pkgs


def all_vector_store_requires():
    """
    pip install "proj[vstore]"
    """
    setup_spec.extras["vstore"] = [
        "grpcio==1.47.5",  # maybe delete it
        "pymilvus==2.2.1",
        "weaviate-client",
    ]


def all_datasource_requires():
    """
    pip install "proj[datasource]"
    """

    setup_spec.extras["datasource"] = [
        "pymssql",
        "pymysql",
        "pyspark",
        "psycopg2",
        # for doris
        # mysqlclient 2.2.x have pkg-config issue on 3.10+
        "mysqlclient==2.1.0",
        "pydoris>=1.0.2,<2.0.0",
        "google-cloud-bigquery"
    ]


def openai_requires():
    """
    pip install "proj[openai]"
    """
    setup_spec.extras["openai"] = ["tiktoken"]
    if BUILD_VERSION_OPENAI:
        # Read openai sdk version from env
        setup_spec.extras["openai"].append(f"openai=={BUILD_VERSION_OPENAI}")
    else:
        setup_spec.extras["openai"].append("openai")

    setup_spec.extras["openai"] += setup_spec.extras["framework"]
    setup_spec.extras["openai"] += setup_spec.extras["knowledge"]


def gpt4all_requires():
    """
    pip install "proj[gpt4all]"
    """
    setup_spec.extras["gpt4all"] = ["gpt4all"]




def cache_requires():
    """
    pip install "proj[cache]"
    """
    setup_spec.extras["cache"] = ["rocksdict"]


def default_requires():
    """
    pip install "proj[default]"
    """
    setup_spec.extras["default"] = [
        # "tokenizers==0.13.3",
        "tokenizers>=0.14",
        "accelerate>=0.20.3",
        "protobuf==3.20.3",
        "zhipuai",
        "dashscope",
        "chardet",
    ]
    setup_spec.extras["default"] += setup_spec.extras["framework"]
    setup_spec.extras["default"] += setup_spec.extras["knowledge"]
    setup_spec.extras["default"] += setup_spec.extras["torch"]
    setup_spec.extras["default"] += setup_spec.extras["quantization"]
    setup_spec.extras["default"] += setup_spec.extras["cache"]


def all_requires():
    requires = set()
    for _, pkgs in setup_spec.extras.items():
        for pkg in pkgs:
            requires.add(pkg)
    setup_spec.extras["all"] = list(requires)


def init_install_requires():
    setup_spec.install_requires += setup_spec.extras["core"]
    print(f"Install requires: \n{','.join(setup_spec.install_requires)}")


core_requires()
torch_requires()
knowledge_requires()
# llama_cpp_requires()
quantization_requires()

all_vector_store_requires()
all_datasource_requires()
openai_requires()
gpt4all_requires()
# vllm_requires()
cache_requires()

# must be last
default_requires()
all_requires()
init_install_requires()

# Packages to exclude when IS_DEV_MODE is False
excluded_packages = ["tests", "*.tests", "*.tests.*", "examples"]

if IS_DEV_MODE:
    packages = find_packages(exclude=excluded_packages)
else:
    packages = find_packages(
        exclude=excluded_packages,
        include=[
            "proj",
            "proj._private",
            "proj._private.*",
            "proj.cli",
            "proj.cli.*",
            "proj.configs",
            "proj.configs.*",
            "proj.core",
            "proj.core.*",
            "proj.util",
            "proj.util.*",
            "proj.model",
            "proj.model.proxy",
            "proj.model.proxy.*",
            "proj.model.operators",
            "proj.model.operators.*",
            "proj.model.utils",
            "proj.model.utils.*",
        ],
    )

setuptools.setup(
    name="proj",
    packages=packages,
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=setup_spec.install_requires,
    python_requires=">=3.10",
    extras_require=setup_spec.extras
)