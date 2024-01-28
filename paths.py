import os

path = os.path.dirname(os.path.realpath(__file__))

def download_location(year):
    return f'{path}/data/{year}/blue_cache_data/'
def teams_location(year):
    return f'{path}/data/{year}/blue_cache_data/teams.csv'

def x_offense_location(year):
    return f'{path}/data/{year}/blue_cache_data/x_offense_location.npy'
def x_defense_location(year):
    return f'{path}/data/{year}/blue_cache_data/x_defense_location.npy'
def x_meta_location(year):
    return f'{path}/data/{year}/blue_cache_data/x_meta_location.npy'
def y_location(year):
    return f'{path}/data/{year}/blue_cache_data/y.npy'

def test_x_offense_location(year):
    return f'{path}/data/{year}/blue_cache_data/test_x_offense_location.npy'
def test_x_defense_location(year):
    return f'{path}/data/{year}/blue_cache_data/test_x_defense_location.npy'
def test_x_meta_location(year):
    return f'{path}/data/{year}/blue_cache_data/test_x_meta_location.npy'
def test_y_location(year):
    return f'{path}/data/{year}/blue_cache_data/test_y.npy'

def model_location(year):
    return f'{path}/data/{year}/model/'
