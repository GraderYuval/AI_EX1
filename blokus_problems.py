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
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
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
    min_distance_to_lt = distance_to_connected(state, (0, 0))  # from (0,0)
    min_distance_to_lb = distance_to_connected(state, (0, state.board_h - 1))  # from (0,h)
    min_distance_to_rt = distance_to_connected(state, (state.board_w - 1, 0))  # from (w,0)
    min_distance_to_rb = distance_to_connected(state, (state.board_w - 1, state.board_h - 1))  # from (w,h)

    uncovered_corrners = (0 if state.get_position(0, 0) != -1 else 1) + \
                         (0 if state.get_position(0, state.board_h - 1) != -1 else 1) + \
                         (0 if state.get_position(state.board_w - 1, 0) != -1 else 1) + \
                         (0 if state.get_position(state.board_w - 1, state.board_h - 1) != -1 else 1) - 1

    max_min_dist = max(min_distance_to_lt, min_distance_to_lb, min_distance_to_rt, min_distance_to_rb)
    return max_min_dist + uncovered_corrners


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def set_board(self, board):
        self.board = board

    def is_goal_state(self, state):
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
        cost = 0
        for action in actions:
            cost += action.piece.num_tiles
        return cost


def blokus_cover_heuristic(state, problem):
    max_dist = 0
    for t in problem.targets:
        curr_closest_point_max_dist = distance_to_connected(state, t)
        if max_dist < curr_closest_point_max_dist:
            max_dist = curr_closest_point_max_dist

    return max_dist


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

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_closest_target_idx(self, target_found):
        min_target_idx = -1
        min_distance = float('inf')
        for target_idx, found in enumerate(target_found):
            if found:
                continue
            distance = distance_to_connected(self.board, self.targets[target_idx])
            if distance < min_distance:
                min_distance = distance
                min_target_idx = target_idx
        return min_target_idx

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
        board_copy = self.board.__copy__()
        backtrace = []
        target_found = np.zeros(len(self.targets))  # boolean array indicating which goal was achieved

        while 0 in target_found:
            closest_target_idx = self.get_closest_target_idx(target_found)
            closest_target = self.targets[closest_target_idx]
            problem = BlokusCoverProblem(board_copy.board_w, board_copy.board_h, board_copy.piece_list, self.starting_point,
                                         [closest_target])
            problem.set_board(board_copy)
            actions = ucs(problem)

            backtrace.extend(actions)
            for action in actions:
                board_copy.add_move(0, action)
            target_found[closest_target_idx] = 1
            self.expanded += problem.expanded
        return backtrace

    def legal_state(self, state, totally_forbidden_points):
        for x, y in totally_forbidden_points:
            if state.get_position(x, y) != -1:
                return False
        return True

    def get_next_action(self, successors, target, totally_forbidden_points):
        min_distance = float('inf')
        min_action = None
        for state, action, cost in successors:
            distance = distance_to_connected(state, target)
            if distance < min_distance and self.legal_state(state, totally_forbidden_points):
                min_distance = distance
                min_action = action
                if min_distance == 0:
                    self.legal_state(state, totally_forbidden_points)
                    break
        return min_action

    def get_forbidden_points(self):
        mat = np.zeros((self.board.board_w, self.board.board_h))
        for x, y in self.targets:
            for x_pos, y_pos in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if x_pos < 0 or x_pos >= self.board.board_w or y_pos < 0 or y_pos >= self.board.board_h:
                    continue
                mat[x_pos, y_pos] += 1
        totally_forbidden_points = np.where(mat > 1)
        return list(zip(totally_forbidden_points[0], totally_forbidden_points[1]))


def distance_to_connected(board, target):
    target_x, target_y = target
    if board.get_position(target_x, target_y) != -1:
        return 0
    min_distance = float('inf')
    for x in range(board.board_w):
        for y in range(board.board_h):
            if board.connected[0][x, y] == True:
                cur_distance = max(abs(x - target_x), abs(y - target_y))
                min_distance = min(min_distance, cur_distance)
    return min_distance + 1


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        util.raiseNotDefined()
