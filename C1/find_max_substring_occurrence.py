def find_max_substring_occurrence(input_string):
    substrings = [input_string[0:j] for j in range(1, len(input_string) + 1)]
    k = 0
    for t in substrings:
        count = input_string.count(t)
        if len(t) * count == len(input_string) and count > k:
            k = count
    return k
