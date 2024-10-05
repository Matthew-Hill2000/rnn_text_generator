from setuptools import setup, find_packages

setup(
    name="rnn_text_generator",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.5",
        "tensorboard>=2.6.0",
    ],
    entry_points={
        'console_scripts': [
            'run_rnn_generator=scripts.run:run',
        ],
    },
    python_requires=">=3.7",
    author="Matthew Hill",
    author_email="mattiejhill@gmail.com@example.com",
    description="A RNN-based text generator",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Matthew-Hill2000/rnn_text_generator",
)
