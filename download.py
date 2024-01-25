import tensorflow as tf
import numpy as np

import tbapy
import threading
import os

import shared
import config
import util

tba = tbapy.TBA('3gnerr3ePmpTujuPLT79EyIr0xHC3fSzZBhdmg8EOZSM2nY0duhvb6oYbxx4yimU')
download_batch_size = 12

def is_normal(event_type):
    return event_type == shared.regional or \
        event_type == shared.district or \
        event_type == shared.district_championship or \
        event_type == shared.district_championship_division


def is_champ(event_type):
    return event_type == shared.championship_division or \
        event_type == shared.championship_final


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


def get_event_matches(key, meta):
    week, = meta
    raw_matches = tba.event_matches(key)

    matches = []
    for raw_match in raw_matches:
        try:
            blue = raw_match['alliances']['blue']
            red = raw_match['alliances']['red']
            matches.append(((blue['team_keys'], (blue['score'])), 
                            (red['team_keys'], (red['score'])),
                            (raw_match['comp_level'] != 'qm', week)))
        except:
            pass

    return matches


def match_to_data(match, teams_to_ids, on_hot_teams):
    (blue_teams, blue_score), (red_teams, red_score), meta = match

    one_hot_blue = sum([on_hot_teams[teams_to_ids[blue_team]] for blue_team in blue_teams])
    one_hot_red = sum([on_hot_teams[teams_to_ids[red_team]] for red_team in red_teams])
    meta = tf.constant(meta, dtype=tf.float32)

    blue_y = tf.constant([blue_score], dtype=tf.float32)
    red_y = tf.constant([red_score], dtype=tf.float32)

    return one_hot_blue, one_hot_red, meta, blue_y, red_y


def load(key, meta, teams, matches):
    teams.update(get_event_teams(key))
    matches.extend(get_event_matches(key, meta))


def load_keys(keys_and_metas):
    teams = set()
    matches = []

    batches = [keys_and_metas[i:i + download_batch_size] for i in range(0, len(keys_and_metas), download_batch_size)]
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

os.makedirs(shared.download_location, exist_ok=True)

with open(shared.teams_location, 'w+') as file:
    file.write('\n'.join(teams))

np.save(shared.x_offense_location, x_offense)
np.save(shared.x_defense_location, x_defense)
np.save(shared.x_meta_location, x_meta)
np.save(shared.y_location, y)

np.save(shared.test_x_offense_location, test_x_offense)
np.save(shared.test_x_defense_location, test_x_defense)
np.save(shared.test_x_meta_location, test_x_meta)
np.save(shared.test_y_location, test_y)
