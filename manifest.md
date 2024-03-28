# Diamond value prediction
This brief documents explains in a preliminary way the ecosystem employed to develop the system and the motivations behind it. Overall, the task requires the end-to-end development of a ML solution for diamond evaluation/pricing. This requires going over the typical data exploration phases, model development and automation through a pipeline. At any time during development these considerations may change, especially after the information gathered during the data exploration phase.

## Tech stack
**Python** is the ML related technology I am most familiar with, it is an obvious choice for me. Of the existing frameworks for data manipulation, visualization and model development, the choice may be: **pandas**, **matplotlib**, **scikit-learn**.

Given the limited available time span, pandas and scikit-learn make for great prototyping candidates. In a broader view, they are not the most scalable tools out there, but they offer great interoperability, which will come handy during the development (TODO: make some scalability considerations after the data exploration phase).

Eventually, **pytorch** may be used to train a small network or a linear regressor, keeping in mind that explainability is required by the customer. If possible, I will avoid using torch, mostly to keep the dependencies tidy and small, given the relatively small task and data

For quicker visualization in terms of development time and tidier code overall, **seaborn** may be used on top of matplotlib.

Either **Flask** or **FastAPI** will be used to serve the model through a REST API.

A simple pipeline may be developed in Python from scratch. When possible, **pytest** will be employed to unit test the components of the system. Due to time constraints and the intrinsic complications in testing ML models, high coverage may not be achieved.

## Project structure and development cycles
The project will make use of jupyter notebooks for quick and readable development, especially for the first challenges (data analysis, model development). Overall the project will cover all the challenges, but not necessarily in an explicit way (e.g., not necessarily one notebook per challenge, etc.).

Most of the code will however reside in a properly formed python package, which will act as codebase for the entire project, the `diamond` package. Notebooks would then import the package and use the provided functionality. Ideally, all prototyped code will iteratively move from the notebooks to the codebase, leaving only high level scripting and visualization code on the notebooks.
