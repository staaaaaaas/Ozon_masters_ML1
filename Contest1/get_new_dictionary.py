def get_new_dictionary(input_dict_name, output_dict_name):
    dictionary = {}
    with open(input_dict_name, 'r') as f:
        for line in f.readlines()[1:]:
            line_list = line.replace(
                ',', ' -').replace(' ', '').replace('\n', '').split('-')
            for value in line_list[1:]:
                dictionary[value] = dictionary.get(value, []) + [line_list[0]]
    sorted_dictionary = {
        key: sorted(
            dictionary[key]) for key in sorted(dictionary)}
    with open(output_dict_name, 'w') as f:
        print(str(len(sorted_dictionary)), file=f)
        for key, value in sorted_dictionary.items():
            print(key, end=' - ', file=f)
            print(*value, sep=', ', file=f)
