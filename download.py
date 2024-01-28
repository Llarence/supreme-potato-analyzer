import tensorflow as tf
import numpy as np

import tbapy
import threading
import os

from . import paths, util
from .test import config

REGIONAL = 0
DISTRICT = 11
DISTRICT_CHAMPIONSHIP = 2
DISTRICT_CHAMPIONSHIP_DIVISION = 5
CHAMPIONSHIP_DIVISON = 3
CHAMPIONSHIP_FINAL = 4

DOWNLOAD_BATCH_SIZE = 12

with open(paths.key_location) as file:
    key = file.read()

tba = tbapy.TBA(key)

def is_normal(event_type):
    return event_type == REGIONAL or \
        event_type == DISTRICT or \
        event_type == DISTRICT_CHAMPIONSHIP or \
        event_type == DISTRICT_CHAMPIONSHIP_DIVISION


def is_champ(event_type):
    return event_type == CHAMPIONSHIP_DIVISON or \
        event_type == CHAMPIONSHIP_FINAL


def get_event_keys_and_metas(year):
    raw_events = tba.events(year)

    normal_keys_and_metas = []
    champ_keys_and_metas = []
    for raw_event in raw_events:
        event_type = raw_event['event_type']
        if is_normal(event_type):
            normal_keys_and_metas.append((raw_event['key'], (raw_event['week'],)))
        elif is_champ(event_type):
            champ_keys_and_metas.append((raw_event['key'], (6,)))

    return normal_keys_and_metas, champ_keys_and_metas


def get_event_teams(key):
    raw_teams = tba.event_teams(key)

    teams = []
    for raw_team in raw_teams:
        teams.append(raw_team['key'])

    return teams


def get_points(breakdown):
    return (breakdown['totalPoints'], breakdown['autoPoints'], breakdown['teleopPoints'], breakdown['foulPoints'])


def get_event_matches(key, meta):
    week, = meta
    raw_matches = tba.event_matches(key)

    matches = []
    for raw_match in raw_matches:
        try:
            alliances = raw_match['alliances']
            score_breakdown = raw_match['score_breakdown']
            blue_score = score_breakdown['blue']
            red_score = score_breakdown['red']
            matches.append(((alliances['blue']['team_keys'], get_points(blue_score)), 
                            (alliances['red']['team_keys'], get_points(red_score)),
                            (raw_match['comp_level'] != 'qm', week)))
        except:
            pass

    return matches


def match_to_data(match, teams_to_ids, on_hot_teams):
    (blue_teams, blue_score), (red_teams, red_score), meta = match

    one_hot_blue = sum([on_hot_teams[teams_to_ids[blue_team]] for blue_team in blue_teams])
    one_hot_red = sum([on_hot_teams[teams_to_ids[red_team]] for red_team in red_teams])
    meta = tf.constant(meta, dtype=tf.float32)

    blue_y = tf.constant(blue_score, dtype=tf.float32)
    red_y = tf.constant(red_score, dtype=tf.float32)

    return one_hot_blue, one_hot_red, meta, blue_y, red_y


def load(key, meta, teams, matches):
    teams.update(get_event_teams(key))
    matches.extend(get_event_matches(key, meta))


def load_keys(keys_and_metas):
    teams = set()
    matches = []

    batches = [keys_and_metas[i:i + DOWNLOAD_BATCH_SIZE] for i in range(0, len(keys_and_metas), DOWNLOAD_BATCH_SIZE)]
    for curr_keys_and_metas in util.show_percent(batches):
        threads = []
        for key, meta in curr_keys_and_metas:
            thread = threading.Thread(target=load, args=(key, meta, teams, matches))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
    
    return matches, teams


def matches_to_data(matches, teams_to_ids, one_hot_teams):
    x_offense = []
    x_defense = []
    x_meta = []
    y = []
    for match in util.show_percent(matches):
        blue_x, red_x, meta_x, blue_y, red_y = match_to_data(match, teams_to_ids, one_hot_teams)

        x_offense.append(blue_x)
        x_defense.append(red_x)
        x_meta.append(meta_x)
        y.append(blue_y)

        x_offense.append(red_x)
        x_defense.append(blue_x)
        x_meta.append(meta_x)
        y.append(red_y)

    x_offense = tf.stack(x_offense)
    x_defense = tf.stack(x_defense)
    x_meta = tf.stack(x_meta)
    y = tf.stack(y)

    return x_offense, x_defense, x_meta, y


def download(year):
    normal_keys_and_metas, champ_keys_and_metas = get_event_keys_and_metas(config.year)

    normal_matches, teams1 = load_keys(normal_keys_and_metas)
    champ_matches, teams2 = load_keys(champ_keys_and_metas)

    teams = list(teams1.union(teams2))
    num_teams = len(teams)

    team_ids = range(num_teams)
    teams_to_ids = dict(zip(teams, team_ids))

    one_hot_teams = tf.one_hot(team_ids, num_teams)

    x_offense, x_defense, x_meta, y = matches_to_data(normal_matches, teams_to_ids, one_hot_teams)
    test_x_offense, test_x_defense, test_x_meta, test_y = matches_to_data(champ_matches, teams_to_ids, one_hot_teams)

    os.makedirs(paths.download_location(year), exist_ok=True)

    with open(paths.teams_location(year), 'w+') as file:
        file.write('\n'.join(teams))

    np.save(paths.x_offense_location(year), x_offense)
    np.save(paths.x_defense_location(year), x_defense)
    np.save(paths.x_meta_location(year), x_meta)
    np.save(paths.y_location(year), y)

    np.save(paths.test_x_offense_location(year), test_x_offense)
    np.save(paths.test_x_defense_location(year), test_x_defense)
    np.save(paths.test_x_meta_location(year), test_x_meta)
    np.save(paths.test_y_location(year), test_y)


if __name__ == '__main__':
    download(config.year)
