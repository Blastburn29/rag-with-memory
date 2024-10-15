# This repository is about RAG Application development

In `part1/SampleSetAssessment.ipynb` contains the main backend logic for the project. 
It demonstrates how the application works and the driver logic code is also in the same file. Since it is a jupyter notebook, required dependencies are mentioned in the book itself

In `part2`, I have created a streamlit based frontend to host my RAG application. This application is hosted on streamlit cloud and this directory contains the streamlit and `part1` code too.

## In order to use part2 directly with python and streamlit,

* Clone the repository from part 2

* Install the required libraries

```bash
pip install -r requirements.txt
```
* Run Streamlit command

```bash
streamlit run app.py
```

---
# Part 2 Docker Initialized
Part 2 contains a `Dockerfile`. To run streamlit app on docker.

* Step 1: Create a docker image
```bash
docker build . -t image_name
```
* Step 2: Check the docker image list to verify the image has been created sucessfully
```bash
docker image ls
```
* Step 3: Create and port Docker container towards streamlit's default port

```bash
docker run -p 8080:8501 <docker image_id>
```



## Examples for Part 2

![Part 2 Example 1](assests/image.png)

![Part 2 Example 2](assests/image-1.png)

![Part 2 Example 3](assests/image-2.png)