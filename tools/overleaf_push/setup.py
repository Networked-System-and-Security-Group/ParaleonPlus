from setuptools import setup


setup(
    name="overleaf-helper",
    version="0.1.0",
    py_modules=["overleaf_push"],
    install_requires=["requests>=2.31,<3"],
    entry_points={
        "console_scripts": [
            "overleaf-helper=overleaf_push:main",
        ]
    },
)
