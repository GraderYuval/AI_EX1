from board import Board
from search import SearchProblem, ucs
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

                # min_distance_to_lt2 = min(min_distance_to_lt2, min(x, y))  # from (0,0)
                # min_distance_to_lb2 = min(min_distance_to_lb2, min(x, state.board_h - y))  # from (0,h)
                # min_distance_to_rt2 = min(min_distance_to_rt2, min(state.board_w - x, y))  # from (w,0)
                # min_distance_to_rb2 = min(min_distance_to_rb2, min(state.board_w - x, state.board_h - y))  # from (w,h)

    max_min_dist = max(min_distance_to_lt, min_distance_to_lb, min_distance_to_rt, min_distance_to_rb)
    return max_min_dist
    #
    # mapping = {0: min_distance_to_lt2, 1: min_distance_to_lb2, 2: min_distance_to_rt2, 3: min_distance_to_rb2}
    #
    # as_arr = np.asarray([min_distance_to_lt, min_distance_to_lb, min_distance_to_rt, min_distance_to_rb])
    # arg_max = np.argmax(as_arr)
    # sum_min_dist = min_distance_to_lt2 + min_distance_to_lb2 + min_distance_to_rt2 + min_distance_to_rb2
    # sum_min_dist -= mapping[arg_max]
    # return max_min_dist + sum_min_dist


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


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    # l = left, t = top, r = right, b = bottom

    next_direction = {'u': 'l', 'l': 'd', 'd': 'r', 'r': 'u'}
    max_min_distance = 0
    for t_x, t_y in problem.targets:
        cycle_num = 0
        direction = 'u'
        step_size = 1
        x, y = t_x, t_y
        found = False
        if state.get_position(x, y) != -1:
            found = True
        while not found:
            base_x, base_y = x, y
            for i in range(1, step_size + 1):
                if direction == 'u':
                    y = base_y - i
                elif direction == 'l':
                    x = base_x - i
                elif direction == 'd':
                    y = base_y + i
                elif direction == 'r':
                    x = base_x + i
                if x >= 0 and x >= state.board_w and y < 0 and y >= state.board_h and state.get_position(x, y) != -1:
                    found = True
            if not found:
                direction = next_direction[direction]
                if cycle_num == 1:
                    step_size += 1
                cycle_num = 1 - cycle_num
        min_distance_to_target = max(abs(x - t_x), abs(y - t_y))
        max_min_distance = min(max_min_distance, min_distance_to_target)
    return max_min_distance


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_closest_target(self):
        target_idx = 0
        return target_idx

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
        pass
        # current_state = self.board.__copy__()
        # backtrace = []
        # target_found = np.zeros(len(self.targets))
        # while 0 in target_found:
        #     target_idx = self.get_closest_target()
        #     #find path to target (add actions to backtrace)
        #     #mark target as visited
        #     backtrace.extend(actions)
        #
        # return backtrace


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
