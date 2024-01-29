import os

path = os.path.dirname(os.path.realpath(__file__))

key_location = f'{path}/key.txt'

def download_location(year):
    return f'{path}/data/{year}/download/'
def teams_location(year):
    return f'{path}/data/{year}/download/teams.csv'

def x_offense_location(year):
    return f'{path}/data/{year}/download/x_offense_location.npy'
def x_defense_location(year):
    return f'{path}/data/{year}/download/x_defense_location.npy'
def x_meta_location(year):
    return f'{path}/data/{year}/download/x_meta_location.npy'
def y_location(year):
    return f'{path}/data/{year}/download/y.npy'

def test_x_offense_location(year):
    return f'{path}/data/{year}/download/test_x_offense_location.npy'
def test_x_defense_location(year):
    return f'{path}/data/{year}/download/test_x_defense_location.npy'
def test_x_meta_location(year):
    return f'{path}/data/{year}/download/test_x_meta_location.npy'
def test_y_location(year):
    return f'{path}/data/{year}/download/test_y.npy'

def model_location(year):
    return f'{path}/data/{year}/model/'
