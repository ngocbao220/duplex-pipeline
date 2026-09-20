from setuptools import setup, find_packages

setup(
    name="duplex-pipelines",
    version="0.1.0",
    packages=find_packages(where=".") + ["cholimex", "duplexchat", "sommelier"],
    package_dir={
        "": ".",
        "cholimex": "pipeline/cholimex/src/cholimex",
        "duplexchat": "pipeline/duplexchat/src/duplexchat",
        "sommelier": "pipeline/sommelier/src/sommelier",
    },
)
