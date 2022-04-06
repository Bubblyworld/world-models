## Installation

To install the Hinterland extension for Jupyter notebooks, which gives you 
nice features like autocompletion, run the following command:

`poetry run jupyter contrib nbextension install --sys-prefix`

To start the Jupyter notebook server, run the following command:

`poetry run jupyter-notebook --notebook-dir="$(pwd)/notebooks"`

Once you have installed the Hinterland extension, the Jupyter notebook webpage will contain a tab titled `Nbextensions`, where you can enable and disable whichever features of the tool you like.
