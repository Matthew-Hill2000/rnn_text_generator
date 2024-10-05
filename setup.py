from setuptools import setup, find_packages

setup(
    name="language_model",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy"
        "tensorboard"
    ],
    entry_points={
        'console_scripts': [
            'rnn_text_generator = scripts/run:run',
        ],
    },
    author='Matthew Hill',
    description='RNN Text Generator for sequence prediction tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Matthew-Hill2000/rnn_text_generator',  # GitHub URL
)