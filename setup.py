import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vidmodex",
    version="0.9a",
    author="Somnath Kumar",
    author_email="hexplex0xff@gmail.com",
    description="Code for Model Extration for Video Classification for Kinetics 400/600 trained model",
   # long_description=long_description,
    url="https://github.com/hex-plex/vidmodex",
    packages=setuptools.find_packages(),
    install_requires=['mmaction2','timm','tqdm','torch','moviepy']
)
