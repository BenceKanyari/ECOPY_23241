def contains_values(input_list, element):
    return element in input_list


def number_of_elements_in_list(input_list):
    return len(input_list)


def remove_every_element_from_list(input_list):
    input_list.clear()


def reverse_list(input_list):
    return input_list[::-1]


def odds_from_list(input_list):
    odd_elements = [x for x in input_list if x % 2 != 0]
    return odd_elements


def number_of_odds_in_list(input_list):
    odd_count = sum(1 for x in input_list if x % 2 != 0)
    return odd_count


def contains_odd(input_list):

    for element in input_list:
        if element % 2 != 0:
            return True

    return False


def second_largest_in_list(input_list):
    if len(input_list) < 2:
        raise ValueError("The list does not contain at least two items")

    sorted_list = sorted(input_list, reverse=True)
    return sorted_list[1]


def sum_of_elements_in_list(input_list):
    return float(sum(input_list))


def cumsum_list(input_list):
    cumsum = []
    current_sum = 0

    for element in input_list:
        current_sum += element
        cumsum.append(current_sum)

    return cumsum


def element_wise_sum(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("The two lists are not the same length")

    result = [x + y for x, y in zip(input_list1, input_list2)]
    return result


def subset_of_list(input_list, start_index, end_index):
    if start_index < 0 or end_index >= len(input_list) or start_index > end_index:
        raise ValueError("Invalid indices")

    subset = input_list[start_index:end_index + 1]
    return subset


def every_nth(input_list, step_size):
    if step_size < 1:
        raise ValueError("Invalid step spacing")

    result = input_list[::step_size]
    return result


def only_unique_in_list(input_list):
    unique_set = set(input_list)
    return len(unique_set) == len(input_list)


def keep_unique(input_list):
    unique_list = []

    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)

    return unique_list


def swap(input_list, first_index, second_index):
    if (
        first_index < 0
        or second_index < 0
        or first_index >= len(input_list)
        or second_index >= len(input_list)
    ):
        raise ValueError("Invalid indices")

    modified_list = input_list.copy()
    modified_list[first_index], modified_list[second_index] = (
        modified_list[second_index],
        modified_list[first_index],
    )
    return modified_list


def remove_element_by_value(input_list, value_to_remove):
    modified_list = [x for x in input_list if x != value_to_remove]
    return modified_list


def remove_element_by_index(input_list, index):
    if index < 0 or index >= len(input_list):
        raise ValueError("Invalid indices")

    modified_list = input_list[:index] + input_list[index + 1:]
    return modified_list


def multiply_every_element(input_list, multiplier):
    modified_list = [x * multiplier for x in input_list]
    return modified_list


def remove_key(input_dict, key):
    if key in input_dict:
        del input_dict[key]
    return input_dict


def sort_by_key(input_dict):
    sorted_items = sorted(input_dict.items())
    sorted_dict = dict(sorted_items)
    return sorted_dict


def sum_in_dict(input_dict):
    total_sum = sum(input_dict.values())
    return total_sum


def merge_two_dicts(input_dict1, input_dict2):
    merged_dict = {**input_dict1, **input_dict2}
    return merged_dict


def merge_dicts(*dicts):
    merged_dict = {}

    for d in dicts:
        merged_dict.update(d)

    return merged_dict


def sort_list_by_parity(input_list):
    even_numbers = [x for x in input_list if x % 2 == 0]
    odd_numbers = [x for x in input_list if x % 2 != 0]
    return {'even': even_numbers, 'odd': odd_numbers}


def mean_by_key_value(input_dict):
    result_dict = {}

    for key, value_list in input_dict.items():
        if len(value_list) > 0:
            mean_value = sum(value_list) / len(value_list)
            result_dict[key] = mean_value
        else:
            result_dict[key] = None

    return result_dict


def count_frequency(input_list):
    frequency_dict = {}

    for item in input_list:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1

    return frequency_dict
