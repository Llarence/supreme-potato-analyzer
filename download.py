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

def get_event_keys(year):
    raw_events = tba.events(year)

    keys = []
    for raw_event in raw_events:
        if raw_event['event_type'] != shared.offseason_type and raw_event['event_type'] != shared.offseason_type:
            keys.append(raw_event['key'])

    return keys


def get_event_teams(key):
    raw_teams = tba.event_teams(key)

    teams = []
    for raw_team in raw_teams:
        teams.append(raw_team['key'])

    return teams


def get_event_matches(key):
    raw_matches = tba.event_matches(key)

    matches = []
    for raw_match in raw_matches:
        try:
            blue = raw_match['alliances']['blue']
            red = raw_match['alliances']['red']
            matches.append(((blue['team_keys'], blue['score']), 
                            (red['team_keys'], red['score']),
                            (float(raw_match['comp_level'] != 'qm'),)))
        except:
            pass

    return matches


def match_to_tensor(match, teams_to_ids, on_hot_teams):
    (blue_teams, blue_score), (red_teams, red_score), meta = match

    one_hot_blue = sum([on_hot_teams[teams_to_ids[blue_team]] for blue_team in blue_teams])
    one_hot_red = sum([on_hot_teams[teams_to_ids[red_team]] for red_team in red_teams])
    meta = tf.constant(meta)

    blue_data = (tf.concat([one_hot_blue, one_hot_red, meta], 0), tf.constant(blue_score))
    red_data = (tf.concat([one_hot_red, one_hot_blue, meta], 0), tf.constant(red_score))
    return blue_data, red_data


keys = get_event_keys(config.year)

teams = set()
matches = []

def load(key):
    teams.update(get_event_teams(key))
    matches.extend(get_event_matches(key))


batches = [keys[i:i + download_batch_size] for i in range(0, len(keys), download_batch_size)]
for curr_keys in util.show_percent(batches):
    threads = []
    for key_num, key  in enumerate(curr_keys):
        thread = threading.Thread(target=load, args=(key,))
        threads.append(thread)
        thread.start()

    for thread_num, thread in enumerate(threads):
        thread.join()

teams = list(teams)
num_teams = len(teams)

team_ids = range(num_teams)
teams_to_ids = dict(zip(teams, team_ids))

one_hot_teams = tf.one_hot(team_ids, num_teams)

x = []
y = []
for match in util.show_percent(matches):
    (blue_x, blue_y), (red_x, red_y) = match_to_tensor(match, teams_to_ids, one_hot_teams)

    x.append(blue_x)
    x.append(red_x)

    y.append(blue_y)
    y.append(red_y)

x = tf.stack(x)
y = tf.stack(y)

os.makedirs(shared.download_location, exist_ok=True)

with open(shared.teams_location, 'w+') as file:
    file.write('\n'.join(teams))

np.save(shared.xs_location, x)
np.save(shared.ys_location, y)
