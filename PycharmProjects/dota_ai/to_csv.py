import csv

# Открываем файл .txt для чтения
with open('123122.txt', 'r') as txtfile:
    lines = txtfile.readlines()

    # Открываем файл .csv для записи
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for line in lines:
            # Разделяем строку по точке с запятой и записываем в файл .csv
            writer.writerow(line.strip().split(';'))