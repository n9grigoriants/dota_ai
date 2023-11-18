import numpy as np

# список героев
heroes_list = ['kunkka', 'pudge', 'io', 'jakiro', 'juggernaut', 'meepo', 'bounty_hunter', 'lion', 'phoenix', 'abaddon']
items_dict = {
    'dagon': 1,
    'battle_fury': 2,
    'boots_of_speed': 3,
    'bottle': 4,
    'ward': 5,
    'magic_wand': 6,
    # + остальные предметы
}

# создание словаря с рандомными значениями для каждого героя
heroes_dict = {}
for hero in heroes_list:
    # генерация рандомного вектора из 5 чисел в диапазоне [0, 1] с округлением до тысячных
    random_vector = [round(x, 3) for x in np.random.rand(5)]
    heroes_dict[hero] = random_vector


def convert_input_to_matrix(input_data):
    matrix = []

    # индикатор принадлежности команде
    team_indicator = [0, 1, 1, 1, 1, -1, -1, -1, -1, -1]

    # преобразование героев в векторы
    for i, hero_name in enumerate(input_data[1:11]):
        hero_vector = [team_indicator[i]] + heroes_dict[hero_name]
        matrix.append(hero_vector)

    # последний элемент входных данных - список предметов
    items_list = [items_dict[item] for item in input_data[-1]]

    return matrix, items_list


input_data1 = [1, 'kunkka', 'pudge', 'io', 'jakiro', 'juggernaut', 'meepo', 'bounty_hunter', 'lion', 'phoenix', 'abaddon', ['dagon', 'battle_fury', 'boots_of_speed', 'bottle', 'ward', 'magic_wand']]  # ваш первый входной список

matrix1, items1 = convert_input_to_matrix(input_data1)

print(matrix1)
print(items1)
