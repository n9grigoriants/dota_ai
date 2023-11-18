import requests
import json
from random import randint
from time import sleep

with open('heroes.json', 'r') as heroes:
    heroes = json.load(heroes)
with open('items_upd.json', 'r') as items:
    items = json.load(items)

heroes = {hero_data['id']: hero_data['localized_name'].lower().replace(' ', '_').replace("'", "") for hero_data in
          heroes.values()}
items = {item_data['id']: item_data['dname'].lower().replace(' ', '_').replace("'", "") for item_data in items.values()
         if 'dname' in item_data}

match_num = 0
for i in range(1000):
    try:
        matches = requests.get(
            f'https://api.opendota.com/api/publicMatches?less_than_match_id={"7411" + str(randint(100000, 999999))}').json()
        match_ids = [match['match_id'] for match in matches]
        for match_id in match_ids:
            match_data = requests.get(f'https://api.opendota.com/api/matches/{match_id}').json()
            try:
                for player in match_data['players']:
                    if player['win'] != 1:
                        continue
                    hero_winner = heroes[player['hero_id']]
                    winner_team = [heroes[player['hero_id']] for player in match_data['players'] if
                                   heroes[player['hero_id']] != hero_winner and player['win'] == 1]
                    loser_team = [heroes[player['hero_id']] for player in match_data['players'] if player['win'] == 0]
                    items_of_hero = []
                    for j in range(6):
                        try:
                            items_of_hero.append(items[player[f'item_{j}']])
                        except:
                            pass
                    match_num += 1
                    winner_team = ','.join(winner_team)
                    loser_team = ','.join(loser_team)
                    items_of_hero = ','.join(items_of_hero)
                    str_to_add = f'{match_num},{hero_winner},{winner_team},{loser_team},"""{items_of_hero}"""'
                    print(str_to_add)
                    with open('../../../Downloads/Telegram Desktop/data.csv', 'a') as data:
                        data.write(str_to_add + '\n')
            except:
                pass
            sleep(1)
    except:
        pass
