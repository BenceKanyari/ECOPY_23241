# NHECBE

def evens_from_list(input_list):
    even_elements = [x for x in input_list if x % 2 == 0]
    return even_elements


def every_element_is_odd(input_list):
    odd_count = sum(1 for x in input_list if x % 2 != 0)
    return odd_count == len(input_list)


def kth_largest_in_list(input_list, kth_largest):
    sorted_list = sorted(input_list, reverse=True)
    return sorted_list[kth_largest - 1]


def cumavg_list(input_list):
    cumavg = []
    current_sum = 0
    n = 0

    for element in input_list:
        n += 1
        current_sum += element
        avg = current_sum / n
        cumavg.append(avg)

    return cumavg


def element_wise_multiplication(input_list1, input_list2):
    result = [x * y for x, y in zip(input_list1, input_list2)]
    return result


def merge_lists(*lists):
    merged_list = []

    for l in lists:
        merged_list.extend(l)

    return merged_list


def squared_odds(input_list):
    return [x * x for x in input_list if x % 2 != 0]


def reverse_sort_by_key(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), reverse = True))
    return sorted_dict


def sort_list_by_divisibility(input_list):
    by_two = [x for x in input_list if x % 2 == 0 and x % 5 != 0]
    by_five = [x for x in input_list if x % 5 == 0 and x % 2 != 0]
    by_two_and_five = [x for x in input_list if x % 2 == 0 and x % 5 == 0]
    by_none = [x for x in input_list if x % 2 != 0 and x % 5 != 0]
    return {'by_two': by_two, 'by_five': by_five, 'by_two_and_five': by_two_and_five, 'by_none': by_none}