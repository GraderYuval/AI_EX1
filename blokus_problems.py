from board import Board
from search import SearchProblem, ucs, depth_first_search_priority
import numpy as np
import util


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return state.get_position(0, 0) != -1 \
               and state.get_position(0, state.board_h - 1) != -1 \
               and state.get_position(state.board_w - 1, 0) != -1 \
               and state.get_position(state.board_w - 1, state.board_h - 1) != -1

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        cost = 0
        for action in actions:
            cost += action.piece.num_tiles
        return cost


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    # l = left, t = top, r = right, b = bottom

    min_distance_to_lt = min_distance_to_lb = min_distance_to_rt = min_distance_to_rb = max(state.board_w,
                                                                                            state.board_h)
    min_distance_to_lt2 = min_distance_to_lb2 = min_distance_to_rt2 = min_distance_to_rb2 = max(state.board_w,
                                                                                                state.board_h)
    for x in range(state.board_w):
        for y in range(state.board_h):
            if state.get_position(x, y) != -1:
                min_distance_to_lt = min(min_distance_to_lt, max(x, y))  # from (0,0)
                min_distance_to_lb = min(min_distance_to_lb, max(x, state.board_h - y))  # from (0,h)
                min_distance_to_rt = min(min_distance_to_rt, max(state.board_w - x, y))  # from (w,0)
                min_distance_to_rb = min(min_distance_to_rb, max(state.board_w - x, state.board_h - y))  # from (w,h)

                min_distance_to_lt2 = min(min_distance_to_lt2, min(x, y))  # from (0,0)
                min_distance_to_lb2 = min(min_distance_to_lb2, min(x, state.board_h - y))  # from (0,h)
                min_distance_to_rt2 = min(min_distance_to_rt2, min(state.board_w - x, y))  # from (w,0)
                min_distance_to_rb2 = min(min_distance_to_rb2, min(state.board_w - x, state.board_h - y))  # from (w,h)

    uncovered_corrners = (0 if state.get_position(0, 0) != -1 else 1) + \
                         (0 if state.get_position(0, state.board_h - 1) != -1 else 1) + \
                         (0 if state.get_position(state.board_w - 1, 0) != -1 else 1) + \
                         (0 if state.get_position(state.board_w - 1, state.board_h - 1) != -1 else 1) - 1

    # max_min_dist = max(min_distance_to_lt, min_distance_to_lb, min_distance_to_rt, min_distance_to_rb) + (uncovered_corrners - 1)
    max_min_dist = max(min_distance_to_lt, min_distance_to_lb, min_distance_to_rt, min_distance_to_rb)
    # return max_min_dist

    mapping = {0: min_distance_to_lt2, 1: min_distance_to_lb2, 2: min_distance_to_rt2, 3: min_distance_to_rb2}

    as_arr = np.asarray([min_distance_to_lt, min_distance_to_lb, min_distance_to_rt, min_distance_to_rb])
    arg_max = np.argmax(as_arr)

    sum_min_dist = min_distance_to_lt2 + min_distance_to_lb2 + min_distance_to_rt2 + min_distance_to_rb2
    sum_min_dist -= mapping[arg_max]
    # return max_min_dist + max(sum_min_dist, uncovered_corrners)
    return max_min_dist + uncovered_corrners


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.expanded = 0

        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def set_board(self, board):
        self.board = board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        for target in self.targets:
            if state.get_position(target[0], target[1]) == -1:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        cost = 0
        for action in actions:
            cost += action.piece.num_tiles
        return cost


def dist_to_rec(x, y, s_x, s_y, t_x, t_y):
    dist_x = 0
    if s_x <= t_x:  # s_x .... t_x
        if x > t_x:
            dist_x = x - t_x
        if x < s_x:
            dist_x = s_x - x
    else:  # t_x .... s_x
        if x < t_x:
            dist_x = t_x - x
        if x > s_x:
            dist_x = x - s_x

    dist_y = 0
    if s_y <= t_y:  # s_y .... t_y
        if y > t_y:
            dist_y = y - t_y
        if y < s_y:
            dist_y = s_y - y
    else:  # t_y .... s_y
        if y < t_y:
            dist_y = t_y - y
        if y > s_y:
            dist_y = y - s_y

    return max(dist_x, dist_y)


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    # l = left, t = top, r = right, b = bottom
    # next_direction = {'u': 'l', 'l': 'd', 'd': 'r', 'r': 'u'}
    # max_min_distance = 0
    # for t_x, t_y in problem.targets:
    #     cycle_num = 0
    #     direction = 'u'
    #     step_size = 1
    #     x, y = t_x, t_y
    #     found = False
    #     if state.get_position(x, y) != -1:
    #         found = True
    #     while not found:
    #         base_x, base_y = x, y
    #         for i in range(1, step_size + 1):
    #             if direction == 'u':
    #                 y = base_y - i
    #             elif direction == 'l':
    #                 x = base_x - i
    #             elif direction == 'd':
    #                 y = base_y + i
    #             elif direction == 'r':
    #                 x = base_x + i
    #             if x >= 0 and x >= state.board_w and y < 0 and y >= state.board_h and state.get_position(x, y) != -1:
    #                 found = True
    #         if not found:
    #             direction = next_direction[direction]
    #             if cycle_num == 1:
    #                 step_size += 1
    #             cycle_num = 1 - cycle_num
    #     min_distance_to_target = max(abs(x - t_x), abs(y - t_y))
    #     max_min_distance = min(max_min_distance, min_distance_to_target)
    # return max_min_distance
    max_dist = (0, (-1, -1), (-1, -1))
    for t_x, t_y in problem.targets:
        curr_closest_point_max_dist = (float('inf'), (-1, -1), (-1, -1))
        for x in range(state.board_w):
            for y in range(state.board_h):
                if state.get_position(x, y) != -1:
                    long_edge = max(abs(x - t_x), abs(y - t_y))
                    if long_edge < curr_closest_point_max_dist[0]:
                        curr_closest_point_max_dist = (long_edge, (x, y), (t_x, t_y))
        if max_dist[0] < curr_closest_point_max_dist[0]:
            max_dist = curr_closest_point_max_dist

    # source_of_max_edge = curr_closest_point_max_dist[1]
    # target_of_max_edge = curr_closest_point_max_dist[2]
    # max_dist_to_rec = 0
    # targets_out_side_rec = 0
    # for t_x, t_y in problem.targets:
    #     if state.get_position(t_x, t_y) == -1:
    #         d = dist_to_rec(t_x, t_y, source_of_max_edge[0], source_of_max_edge[1],
    #                         target_of_max_edge[0], target_of_max_edge[1])
    #         if d > 0:  # target outside rec
    #             targets_out_side_rec += 1
    #         max_dist_to_rec = max(d, max_dist_to_rec)

    # return max_dist[0] + max(max_dist_to_rec, targets_out_side_rec)
    return max_dist[0]


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.starting_point = starting_point
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def distance(self, source_point, target):
        return max(abs(target[0] - source_point[0]), abs(target[1] - source_point[1]))

    def get_closest_target_idx(self, source_point, target_found):
        min_target_idx = -1
        min_distance = float('inf')
        for target_idx, found in enumerate(target_found):
            if found:
                continue
            distance = self.distance(source_point, self.targets[target_idx])
            if distance < min_distance:
                min_distance = distance
                min_target_idx = target_idx
        return min_target_idx

    def update_board_connected(self, board_copy, source_point):
        board_copy.connected = np.full((board_copy.num_players, board_copy.board_h, board_copy.board_w), False,
                                       np.bool_)
        source_x, source_y = source_point
        if source_x + 1 < board_copy.board_w:
            if source_y + 1 < board_copy.board_h:
                board_copy.connected[0][source_x + 1, source_y + 1] = True
            if source_y - 1 >= 0:
                board_copy.connected[0][source_x + 1, source_y - 1] = True
        if source_x - 1 >= 0:
            if source_y + 1 < board_copy.board_h:
                board_copy.connected[0][source_x - 1, source_y + 1] = True
            if source_y - 1 >= 0:
                board_copy.connected[0][source_x - 1, source_y - 1] = True

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"

        board_copy = self.board.__copy__()
        backtrace = []
        source_point = self.starting_point
        target_found = np.zeros(len(self.targets))  # boolean array indicating which goal was achieved
        while 0 in target_found:
            closest_target_idx = self.get_closest_target_idx(source_point, target_found)
            closest_target = self.targets[closest_target_idx]
            problem = BlokusCoverProblem(board_copy.board_w, board_copy.board_h, board_copy.piece_list, source_point, [closest_target])
            problem.set_board(board_copy)
            actions = depth_first_search_priority(problem, source_point)
            target_found[closest_target_idx] = 1
            source_point = closest_target
            backtrace.extend(actions)
            for action in actions:
                board_copy.add_move(0, action)
            # self.update_board_connected(board_copy, source_point)
        return backtrace


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
