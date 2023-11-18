import pandas as pd

# Создадим пустой DataFrame, в который будем добавлять содержимое всех файлов
combined_df = pd.DataFrame()

# Предположим, что ваши файлы называются file1.csv, file2.csv, ..., file7.csv
file_paths = ['combined_file_3.csv', 'big_data.csv']

# Проходим по всем файлам и добавляем их содержимое в combined_df
for file_path in file_paths:
    # Читаем файл без заголовков, так как они не указаны в примерах
    temp_df = pd.read_csv(file_path, header=None)

    # Если combined_df не пустой, обновим индекс в temp_df
    if not combined_df.empty:
        # Установим новый индекс, который начинается после последнего индекса в combined_df
        temp_df[0] = temp_df[0] + combined_df.iloc[-1, 0]

    # Объединяем с combined_df
    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# Теперь у нас есть объединенный DataFrame, который мы можем сохранить в новый CSV-файл
combined_df.to_csv('combined_file_4.csv', header=False, index=False)