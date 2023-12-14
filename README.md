## Overview
This repository hosts B-HAR, a baseline framework for in depth study of human activity recognition. B-HAR gives to researchers the possibility to evaluate and compare HAR methodologies with a common well-defined workflow.

Users can interact with B-HAR by using the provided APIs, developed using FastAPI.

## Installation
B-HAR was tested with `python3.9`, you can easily install the required packages by using `pip` with the following command:
```
pip install -r requirements.txt
```

## Get started
In order to start using B-HAR you have to follow these steps:
* Clone the repository
* Create a `.env` file with these parameters:
```bash
SECRET_KEY=<secret key>
ALGORITHM=<algorithm>
ACCESS_TOKEN_EXPIRE_MINUTES=<minutes>
```
SECRET_KEY is a string that is used to sign the JWT, to generate a secret key you can use the following command on a terminal:
```bash
openssl rand -hex 32
```
ALGORITHM is a string that identifies the algorithm to use to sign the JWT, for example `HS256`.\
ACCESS_TOKEN_EXPIRE_MINUTES is the number of minutes after which the access token will expire, for example `30`.\
After you created the `.env` file you have to move it inside the `api` folder.
* To upload datasets from kaggle you need to use kaggle's public API and you must first authenticate using an API token. You can find the instructions to do that [here](https://www.kaggle.com/docs/api#authentication) under the `Authentication` section.
* Move inside the `api` folder and run the server with the following command on a terminal:
```bash
python main.py
```
* Go to [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to call the APIs (if you want you can change the IP and the port in the `main.py` file)
* Use the `/register` API to register a new user providing a username and a password
* Use the `Authorize` button to insert the credentials
* Use the `/upload_file` or `/upload_from_kaggle` API to upload a dataset
* Use the `/create_bhar` API to create a new B-HAR object providing the attributes to initialize it
* You can now access all the other APIs to do what you want with your B-HAR object, be sure to read the documentation of every API to understand **how and when** to use it

Every change you make to the B-HAR object is saved in the storage folder so you can access it even after you close the server.\
You can use the models provided in the `models` folder or you can create your own models and upload them using the `/upload_user_model` API.\
In the `example` folder you can find a dataset and some ml models to upload and a json to create a B-HAR object.

## Outputs
You can check the outputs of every API at [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) by clicking on the API you want to check.