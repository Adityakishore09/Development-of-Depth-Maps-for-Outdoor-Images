This zip file contains the codes for the DSE 411/611 Course Project on Development of Outdoors Depth Maps.

We followed two approaches, and provide the codes for the same in two different folders.

The dataset used was a custom collected dataset from Bhopal, India and is not publicly available online yet. You can ask for the same from us.

Steps to run ->
1) Open the folder for the approach you want to try
2) Create an environment using pip: python3 -m venv venv
    This will create an environment with the name 'venv'
3) Activate the environment using source venv/bin/activate and install all the dependencies from the relevant 'requirements' file
    pip install -r requirements_approachx.txt where x is either 1 or 2
4) Use the given bash file to run the codes. Note that due to the codes' structure, the changes that may have to be made in the paths have to be
    made in the codes themselves. Thus, all the changable arguments are provided in the codes themselves. The bash files can be run using
    bash run_approachx.sh where x is either 1 or 2.