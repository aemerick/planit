"""

    A. Emerick

    I have no idea what I'm doing.... lets have some fun.....

    Notes about the Strava API:

    1) CANNOT get data about public athletes via the API that you can
       view on the website (does this mean I can / need to do a web scrape?).
       Looks like API only allows you to get data with user permission
       (so this is fine for the prediction portion of the prohect)

    2) 100 req per 15 min and 1000 per day limit for the APP


"""
import time
import pickle
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from stravalib.client import Client

CLIENT_ID = ''
CLIENT_SECRET = ''
REDIRECT_URL = 'http://localhost:8000/authorized'

app = FastAPI()
client = Client()

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        loaded_object = pickle.load(input)
        return loaded_object


def check_token():
    if time.time() > client.token_expires_at:
        refresh_response = client.refresh_access_token(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, refresh_token=client.refresh_token)
        access_token = refresh_response['access_token']
        refresh_token = refresh_response['refresh_token']
        expires_at = refresh_response['expires_at']
        client.access_token = access_token
        client.refresh_token = refresh_token
        client.token_expires_at = expires_at

@app.get("/")
def read_root():
    authorize_url = client.authorization_url(client_id=CLIENT_ID, redirect_uri=REDIRECT_URL)
    return RedirectResponse(authorize_url)


@app.get("/authorized/")
def get_code(state=None, code=None, scope=None):
    token_response = client.exchange_code_for_token(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, code=code)
    access_token = token_response['access_token']
    refresh_token = token_response['refresh_token']
    expires_at = token_response['expires_at']
    client.access_token = access_token
    client.refresh_token = refresh_token
    client.token_expires_at = expires_at
    save_object(client, 'client.pkl')
    return {"state": state, "code": code, "scope": scope}

try:
    client = load_object('client.pkl')
    check_token()
    athlete = client.get_athlete()
    print("For {id}, I now have an access token {token}".format(id=athlete.id, token=client.access_token))

    # To upload an activity
    # client.upload_activity(activity_file, data_type, name=None, description=None, activity_type=None, private=None, external_id=None)
except FileNotFoundError:
    print("No access token stored yet, visit http://localhost:8000/ to get it")
    print("After visiting that url, a pickle file is stored, run this file again to upload your activity")
