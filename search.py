"""
In search.py, you will implement generic search algorithms
"""

import util
import numpy as np


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost


def get_path(final_state_node):
    path = []
    cur_node = final_state_node
    while cur_node.parent is not None:
        path.append(cur_node.action)
        cur_node = cur_node.parent
    path.reverse()
    return path


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

	print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"

    initial_state_node = Node(problem.get_start_state())
    stack = util.Stack()
    stack.push(initial_state_node)
    visited = set()
    while not stack.isEmpty():
        cur_state_node = stack.pop()
        if cur_state_node.state in visited:
            continue
        visited.add(cur_state_node.state)
        if problem.is_goal_state(cur_state_node.state):
            return get_path(cur_state_node)
        for state, action, cost in problem.get_successors(cur_state_node.state):
            stack.push(Node(state, parent=cur_state_node, action=action))
    return []  # TODO: check what should be returned when no solution is found


def depth_first_search_priority(problem, source_point):
    initial_state_node = Node(problem.get_start_state())
    stack = util.Stack()
    stack.push(initial_state_node)
    visited = set()
    source_x, source_y = source_point
    while not stack.isEmpty():
        cur_state_node = stack.pop()
        if cur_state_node.state in visited:
            continue
        visited.add(cur_state_node.state)
        if problem.is_goal_state(cur_state_node.state):
            return get_path(cur_state_node)
        successors = problem.get_successors(cur_state_node.state)
        successors_connected = []
        successors_disconnected = []
        for state, action, cost in successors:
            isolate_piece_board = np.where(state.state == cur_state_node.state.state, -1, 0)
            touching_points = [(source_x + 1, source_y + 1), (source_x + 1, source_y - 1), (source_x - 1, source_y + 1), (source_x - 1, source_y - 1)]
            appended = False
            for x, y in touching_points:
                    if not appended and isolate_piece_board[x, y]:
                        successors_connected.append((state, action, cost))
                        appended = True
            if not appended:
                successors_disconnected.append((state, action, cost))

        for state, action, cost in successors_disconnected:
            stack.push(Node(state, parent=cur_state_node, action=action))
        for state, action, cost in successors_connected:
            stack.push(Node(state, parent=cur_state_node, action=action))
    return []  # TODO: check what should be returned when no solution is found

def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    initial_state_node = Node(problem.get_start_state())
    queue = util.Queue()
    queue.push(initial_state_node)
    visited = set()
    while not queue.isEmpty():
        cur_state_node = queue.pop()
        if cur_state_node.state in visited:
            continue
        visited.add(cur_state_node.state)
        if problem.is_goal_state(cur_state_node.state):
            return get_path(cur_state_node)
        for state, action, cost in problem.get_successors(cur_state_node.state):
            queue.push(Node(state, parent=cur_state_node, action=action))
    return []  # TODO: check what should be returned when no solution is found


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    return astar(problem, heuristic=null_heuristic)


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    min_cost_to_goal = float("inf")
    visited_nodes = set()
    initial_state_node = Node(problem.get_start_state())
    p_quque = util.PriorityQueue()
    p_quque.push(initial_state_node, 0)

    while not p_quque.isEmpty():
        cur_state_node = p_quque.pop()
        if cur_state_node.state in visited_nodes:
            continue

        if problem.is_goal_state(cur_state_node.state):
            return get_path(cur_state_node)

        visited_nodes.add(cur_state_node.state)

        for state, action, cost in problem.get_successors(cur_state_node.state):
            node_cost = cur_state_node.cost + cost
            h_cost = heuristic(state, problem)
            total_cost = node_cost + h_cost
            node = Node(state, parent=cur_state_node, action=action, cost=node_cost)
            p_quque.push(node, total_cost)

    return []


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
