def find_word_in_circle(circle, word):
    len_word = len(word)
    pos = -1
    direction = 1
    new_circle = circle * len_word
    for i in range(len(new_circle)):
        if new_circle[i:i + len_word] == word:
            pos = i
            break
    if pos != -1:
        return (pos, direction)
    direction = -1
    new_circle_reversed = new_circle[::-1]
    for i in range(len(new_circle_reversed)):
        if new_circle_reversed[i:i + len_word] == word:
            pos = len(circle) - i - 1
            break
    if pos != -1:
        return (pos, direction)
    return -1
