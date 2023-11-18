import h5py

with h5py.File('itog/model_1.h5', 'r') as f:

    # Для более глубокого исследования можно использовать рекурсивную функцию
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))


    f.visititems(print_attrs)