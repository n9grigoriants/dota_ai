import pandas as pd
import pickle

# словарь для предметов
items_dict = {'abyssal_blade': 1, 'aeon_disk': 2, 'aether_lens': 3, 'aghanims_blessing': 4, 'aghanims_scepter': 5,
              'aghanims_shard': 6, 'arcane_blink': 7, 'arcane_boots': 8, 'armlet_of_mordiggian': 9,
              'assault_cuirass': 10, 'band_of_elvenskin': 11, 'battle_fury': 12, 'belt_of_strength': 13,
              'black_king_bar': 14, 'blade_mail': 15, 'blade_of_alacrity': 16, 'blades_of_attack': 17,
              'blight_stone': 18, 'blink_dagger': 19, 'blitz_knuckles': 20, 'blood_grenade': 21, 'bloodstone': 22,
              'bloodthorn': 23, 'boots_of_bearing': 24, 'boots_of_speed': 25, 'boots_of_travel': 26, 'bottle': 27,
              'bracer': 28, 'broadsword': 29, 'buckler': 30, 'butterfly': 31, 'chainmail': 32, 'circlet': 33,
              'clarity': 34, 'claymore': 35, 'cloak': 36, 'cornucopia': 37, 'crimson_guard': 38, 'crown': 39,
              'crystalys': 40, 'daedalus': 41, 'dagon': 42, 'demon_edge': 43, 'desolator': 44, 'diadem': 45,
              'diffusal_blade': 46, 'disperser': 47, 'divine_rapier': 48, 'dragon_lance': 49, 'drum_of_endurance': 50,
              'dust_of_appearance': 51, 'eaglesong': 52, 'echo_sabre': 53, 'enchanted_mango': 54, 'energy_booster': 55,
              'eternal_shroud': 56, 'ethereal_blade': 57, 'euls_scepter_of_divinity': 58, 'eye_of_skadi': 59,
              'faerie_fire': 60, 'falcon_blade': 61, 'fluffy_hat': 62, 'force_staff': 63, 'gauntlets_of_strength': 64,
              'gem_of_true_sight': 65, 'ghost_scepter': 66, 'gleipnir': 67, 'glimmer_cape': 68, 'gloves_of_haste': 69,
              'guardian_greaves': 70, 'hand_of_midas': 71, 'harpoon': 72, 'headdress': 73, 'healing_salve': 74,
              'heart_of_tarrasque': 75, 'heavens_halberd': 76, 'helm_of_iron_will': 77, 'helm_of_the_dominator': 78,
              'helm_of_the_overlord': 79, 'holy_locket': 80, 'hurricane_pike': 81, 'hyperstone': 82,
              'infused_raindrops': 83, 'iron_branch': 84, 'javelin': 85, 'kaya': 86, 'kaya_and_sange': 87,
              'linkens_sphere': 88, 'lotus_orb': 89, 'maelstrom': 90, 'mage_slayer': 91, 'magic_stick': 92,
              'magic_wand': 93, 'manta_style': 94, 'mantle_of_intelligence': 95, 'mask_of_madness': 96,
              'medallion_of_courage': 97, 'mekansm': 98, 'meteor_hammer': 99, 'mithril_hammer': 100, 'mjollnir': 101,
              'monkey_king_bar': 102, 'moon_shard': 103, 'morbid_mask': 104, 'mystic_staff': 105, 'null_talisman': 106,
              'nullifier': 107, 'oblivion_staff': 108, 'observer_and_sentry_wards': 109, 'octarine_core': 110,
              'ogre_axe': 111,
              'orb_of_corrosion': 112, 'orb_of_venom': 113, 'orchid_malevolence': 114, 'overwhelming_blink': 115,
              'pavise': 116, 'perseverance': 117, 'phase_boots': 118, 'phylactery': 119, 'pipe_of_insight': 120,
              'platemail': 121, 'point_booster': 122, 'power_treads': 123, 'quarterstaff': 124, 'quelling_blade': 125,
              'radiance': 126, 'reaver': 127, 'refresher_orb': 128, 'revenants_brooch': 129, 'ring_of_basilius': 130,
              'ring_of_health': 131, 'ring_of_protection': 132, 'ring_of_regen': 133, 'robe_of_the_magi': 134,
              'rod_of_atos': 135, 'sacred_relic': 136, 'sages_mask': 137, 'sange': 138, 'sange_and_yasha': 139,
              'satanic': 140, 'scythe_of_vyse': 141, 'shadow_amulet': 142, 'shadow_blade': 143,
              'shivas_guard': 144, 'silver_edge': 145, 'skull_basher': 146, 'slippers_of_agility': 147,
              'smoke_of_deceit': 148, 'solar_crest': 149, 'soul_booster': 150, 'soul_ring': 151, 'spirit_vessel': 152,
              'staff_of_wizardry': 153, 'swift_blink': 154, 'talisman_of_evasion': 155, 'tango': 156,
              'tranquil_boots': 157, 'ultimate_orb': 158, 'urn_of_shadows': 159, 'vanguard': 160,
              'veil_of_discord': 161, 'vitality_booster': 162, 'vladmirs_offering': 163, 'void_stone': 164,
              'voodoo_mask': 165, 'wind_lace': 166, 'wind_waker': 167, 'witch_blade': 168, 'wraith_band': 169,
              'wraith_pact': 170, 'yasha': 171, 'yasha_and_kaya': 172, 'greater_healing_lotus': 173, 'sentry_ward': 174,
              'observer_ward': 175, 'great_healing_lotus': 176, 'boots_of_travel_2': 177, 'cheese': 178,
              'aegis_of_the_immortal': 179, 'healing_lotus': 180, 'refresher_shard': 181, 'aghanims_blessing_-_roshan': 182, 'block_of_cheese': 183, '': 184}
# словарь для героев
heroes_dict_norm = {"abaddon": 1, "alchemist": 2, "ancient_apparition": 3, "anti-mage": 4, "arc_warden": 5, "axe": 6,
                    "bane": 7, "batrider": 8, "beastmaster": 9, "bloodseeker": 10, "bounty_hunter": 11,
                    "brewmaster": 12, "bristleback": 13, "broodmother": 14, "centaur_warrunner": 15, "chaos_knight": 16,
                    "chen": 17, "clinkz": 18, "clockwerk": 19, "crystal_maiden": 20, "dark_seer": 21, "dark_willow": 22,
                    "dawnbreaker": 23, "dazzle": 24, "death_prophet": 25, "disruptor": 26, "doom": 27,
                    "dragon_knight": 28, "drow_ranger": 29, "earth_spirit": 30, "earthshaker": 31, "elder_titan": 32,
                    "ember_spirit": 33, "enchantress": 34, "enigma": 35, "faceless_void": 36, "grimstroke": 37,
                    "gyrocopter": 38, "hoodwink": 39, "huskar": 40, "invoker": 41, "io": 42, "jakiro": 43,
                    "juggernaut": 44, "keeper_of_the_light": 45, "kunkka": 46, "legion_commander": 47, "leshrac": 48,
                    "lich": 49, "lifestealer": 50, "lina": 51, "lion": 52, "lone_druid": 53, "luna": 54, "lycan": 55,
                    "magnus": 56, "marci": 57, "mars": 58, "medusa": 59, "meepo": 60, "mirana": 61, "monkey_king": 62,
                    "morphling": 63, "muerta": 64, "naga_siren": 65, "natures_prophet": 66, "necrophos": 67,
                    "night_stalker": 68, "nyx_assassin": 69, "ogre_magi": 70, "omniknight": 71, "oracle": 72,
                    "outworld_destroyer": 73, "pangolier": 74, "phantom_assassin": 75, "phantom_lancer": 76,
                    "phoenix": 77, "primal_beast": 78, "puck": 79, "pudge": 80, "pugna": 81, "queen_of_pain": 82,
                    "razor": 83, "riki": 84, "rubick": 85, "sand_king": 86, "shadow_demon": 87, "shadow_fiend": 88,
                    "shadow_shaman": 89, "silencer": 90, "skywrath_mage": 91, "slardar": 92, "slark": 93,
                    "snapfire": 94, "sniper": 95, "spectre": 96, "spirit_breaker": 97, "storm_spirit": 98, "sven": 99,
                    "techies": 100, "templar_assassin": 101, "terrorblade": 102, "tidehunter": 103, "timbersaw": 104,
                    "tinker": 105, "tiny": 106, "treant_protector": 107, "troll_warlord": 108, "tusk": 109,
                    "underlord": 110, "undying": 111, "ursa": 112, "vengeful_spirit": 113, "venomancer": 114,
                    "viper": 115, "visage": 116, "void_spirit": 117, "warlock": 118, "weaver": 119, "windranger": 120,
                    "winter_wyvern": 121, "witch_doctor": 122, "wraith_king": 123, "zeus": 124}


def convert_input_to_matrix(row):
    matrix = []

    # принадлежность к команде
    team_indicator = [0, 1, 1, 1, 1, -1, -1, -1, -1, -1]

    # преобразование имен героев в их числовые идентификаторы
    for i, hero_name in enumerate(row[1:11]):
        hero_number = heroes_dict_norm[hero_name]
        matrix.append([team_indicator[i], hero_number])

    # удаление лишних кавычек из строки с предметами
    items_str = str(row[11]).replace('"', '')

    # разделение строки с предметами на отдельные предметы
    items_list = []
    for item in items_str.split(','):
        stripped_item = item.strip()
        if not stripped_item.endswith("_recipe"):
            item_number = items_dict[stripped_item]
            items_list.append(item_number)

    return matrix, items_list


# чтение CSV файла
data = pd.read_csv('combined_file.csv', quotechar='"')

matrices = []
items_lists = []

# преобразование каждой строки данных
for index, row in data.iterrows():
    matrix, items = convert_input_to_matrix(row)
    result_list = [0] * 183
    for bbb in items:
        if 0 <= bbb - 1 < len(result_list):
            result_list[bbb - 1] = 1
    matrices.append(matrix)
    items_lists.append(result_list)
print(matrices)
# print(items_lists)

# сохранение списков матриц
with open('../ai_update/new_matrices.pkl', 'wb') as f:
    pickle.dump(matrices, f)

# сохранение списков предметов
with open('../ai_update/new_items_lists.pkl', 'wb') as f:
    pickle.dump(items_lists, f)
