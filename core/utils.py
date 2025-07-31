from core.transition_table import *


def lexicographic_sorting(entity):
    """
    Lexicographically order entity. Sorted function orders tuples first by the first element, then by the second, etc.
    If time interval is given backwards e.g. (8, 5) it is transformed to (5, 8).

    :param entity: dict - key: state, value: list of time intervals of specific event
    :return: list with lexicographically ordered time intervals
    """
    return sorted([(*ti, state) if ti[0] <= ti[1] else (ti[1], ti[0], state) for state, time_intervals in entity.items() for ti in time_intervals])


def temporal_relations(ti_1, ti_2, epsilon, max_distance):
    """
    Find out the temporal relation between time intervals ti_1 and ti_2 among 7 Allen's temporal relations.
    It is assumed that ti_1 is lexicographically before or equal to ti_2 (ti_1 <= ti_2).

    :param ti_1: first time interval (A.start, A.end)
    :param ti_2: second time interval (B.start, B.end)
    :param epsilon: maximum amount of time between two events that we consider it as the same time
    :param max_distance: maximum distance between two time intervals that means first one still influences the second
    :return: string - one of 7 possible temporal relations or None if relation is unknown
    """
    A_start, A_end = ti_1
    B_start, B_end = ti_2
    if epsilon < B_start - A_end < max_distance:    # before
        return '<'
    elif abs(B_start - A_end) <= epsilon:   # meets
        return 'm'
    elif B_start - A_start > epsilon and A_end - B_start > epsilon and B_end - A_end > epsilon:     # overlaps
        return 'o'
    elif B_start - A_start > epsilon and A_end - B_end > epsilon:   # contains
        return 'c'
    elif B_start - A_start > epsilon and abs(B_end - A_end) <= epsilon:     # finish by
        return 'f'
    elif abs(B_start - A_start) <= epsilon and abs(B_end - A_end) <= epsilon:   # equal
        return '='
    elif abs(B_start - A_start) <= epsilon and B_end - A_end > epsilon:     # starts
        return 's'
    else:
        # print('Other temporal relation!')
        return None
    

def find_all_possible_extensions():
    pass


def equal_TIRPs():
    pass


if __name__ == "__main__":
    entity_symbols = ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'a', 'k', 'b', 'b', 'l', 'd', 'd']
    tirp_symbols = ['a', 'b', 'd', 'b', 'b', 'd']

    # print(check_symbols_lexicographically(entity_symbols, tirp_symbols))
    # print(vertical_support_symbol(entity_list, 'C', 0.1))

    # from entities import entity_list
    # entity = entity_list[1]
    # plot_entity(entity)