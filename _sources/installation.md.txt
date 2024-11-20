# Installation and dependencies

A fresh, separate virtual environment is highly recommended before installing the package.
This can be done using pip, see, e.g., [this](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
This can help to avoid any dependency conflicts and ensure smooth operation of the
package.

First, check if the virtualenv package is installed. To do this, one can run:
```
pip show virtualenv
```
If the package was not found, it can be installed using:
```
pip install virtualenv
```
To create a new environment, run the following:
```
virtualenv PATH/TO/NEW/ENVIRONMENT
```
To activate the environment (on Windows):
```
PATH/TO/NEW/ENVIRONMENTScripts/activate
```
and on Linux:
```
source PATH/TO/NEW/ENVIRONMENTbin/activate
```

Then, package itself can be installed using pip inside the environment:
```
pip install daplis
```

Alternatively, to start using the package, one can download the whole repo. "requirements.txt" 
lists all packages required for this project to run. One can create 
an environment for this project either using conda or pip following the instruction 
above. Once the new environmnt is activated, run the following to install 
the required packages:
```
cd PATH/TO/GITHUB/CODES/daplis
pip install -r requirements.txt
```
Now, the package can be installed via
```
pip install -e .
```
where '-e' stands for editable: any changes introduced to the package will
instantly become a part of the package and can be used without the need
of reinstalling the whole thing. After that, one can import any function 
from the daplis package:
```
from daplis.functions import sensor_plot, delta_t, fits
```

For conda users, the new environment can be installed using the 'requirements' 
text file directly:
```
conda create --name NEW_ENVIRONMENT_NAME --file /PATH/TO/requirements.txt -c conda-forge
```
To install the package, first, switch to the created environment:
```
conda activate NEW_ENVIRONMENT_NAME
```
and run
```
pip install -e .
```