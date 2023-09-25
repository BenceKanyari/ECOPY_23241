import random


def random_from_list(input_list):
    return random.choice(input_list)


def random_sublist_from_list(input_list, number_of_elements):
    return random.choices(input_list, k = number_of_elements)


def random_from_string(input_string):
    return random.choice(input_string)


def hundred_small_random():
    return [random.random() for _ in range(100)]


def hundred_large_random():
    return [random.randint(10, 1000) for _ in range(100)]


def five_random_number_div_three():
    div_three_list = [x for x in range(9, 1000) if x % 3 == 0]
    return random.sample(div_three_list, 5)


def random_reorder(input_list):
    return random.sample(input_list, len(input_list))



def uniform_one_to_five():
    return random.uniform(1, 6)


