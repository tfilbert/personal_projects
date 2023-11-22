import megaminx as mn
import heapq
import copy
import time
import matplotlib.pyplot as plt

# to store how many nodes were expanded
# this value is reset to 0 everytime astar() is ran
expanded_nodes = 0

class Node:
  def __init__(self, puzzle_state, parent=None, move=None):
    self.state = puzzle_state
    self.parent = parent
    self.move = move
    self.g = 0
    self.h = 0
    self.f = 0

"""
Count the closest distance from each edge piece to its home color
sum these up for every edge pieces and divide by 15, since 15 is the max
number of pieces that can be returned to their home face in one turn
"""
def heuristic(puzzle_state):
  # puzzle state is the megaminx object
  total = 0
  for face_color in puzzle_state.colors:
    piece_colors = puzzle_state.get_piece_colors_on_face(face_color)
    for piece_color in piece_colors:
      total += puzzle_state.face_distances[face_color][piece_color]
  h = total // 15
  return h


"""
Guide: https://saturncloud.io/blog/implementing-the-a-algorithm-in-python-a-stepbystep-guide/#:~:text=The%20A*%20algorithm%20works%20by,that%20have%20already%20been%20visited.
I used the above guide to gain a better understanding on how to implment a* in python
I had to change much of this code to make it work with this specific data, so it is not the same
The general structure of the algo and the data structures are similar though 
"""
def astar(start):
  global expanded_nodes 
  expanded_nodes = 0
  counter = 0
  open_list = []
  closed_list = set()

  heapq.heappush(open_list, (start.f, counter, start))

  while open_list:
    current_cost, current_counter, current_node = heapq.heappop(open_list)
    expanded_nodes += 1
    if current_node.state.is_solved():
      # Goal reached, construct and return the path
      path = []
      nodes = []
      while current_node:
        if current_node.parent != None:
          path.append(current_node.move)
          nodes.append(current_node)
        current_node = current_node.parent

      return path[::-1], nodes[0].g

    # goal not reached, add current_node to list of seen nodes
    closed_list.add(current_node)

    for child in get_children(current_node):
      if child in closed_list:
        continue
      # child has not been visited, visit it
      new_cost = current_node.g + 1
      if child not in open_list:
        counter += 1
        heapq.heappush(open_list, (new_cost + heuristic(child.state), counter,  child))
      elif new_cost < child.g:
        child.g = new_cost
        child.parent = current_node
 

def get_children(node):
    children = []
    parent_minx = node.state
    for color in node.state.colors:
      # global expanded_nodes
      # expanded_nodes += 1
      # save current sate, and rotate counter clockwise
      child_minx = copy.deepcopy(parent_minx)
      child_minx.rotate_counter_clockwise(color)
      # create a new node with this new puzzle state, and the passed node as it's parent
      child = Node(child_minx, node, 'CCW-{}'.format(color))
      child.g = node.g+1
      child.h = heuristic(child.state)
      child.f = child.g + child.h
      children.append(child)

    return children


def main():
  # solve j puzzles for i random clockwise rotations
  expanded_by_depth = []
  moves_scrambled_vs_shortest_path = []
  for i in range(3, 5):
    print(i, 'rotations:')
    for j in range(3):
      st = time.time()
      minx = mn.Megaminx()
      minx.random_move(i, 'cw')
      start = Node(minx)
      moves, depth = astar(start)
      global expanded_nodes
      expanded_by_depth.append((depth, expanded_nodes))
      moves_scrambled_vs_shortest_path.append((i, depth))
      print(moves)
      print(time.time()-st)
      # # uncomment to see list of moves made and to print the megaminx after making each move
      # print('Initial state')
      # print(minx.print_megaminx(4))
      # for move in moves:
      #   print('Move made:', move)
      #   minx.rotate_counter_clockwise(move.split('-')[1])
      #   minx.print_megaminx(4)

  # save average expansions per depth
  expansions = {}
  for key, value in expanded_by_depth:
    if key in expansions:
      # If the key already exists in the dictionary, add the value to the list
      expansions[key].append(value)
    else:
      # If the key doesn't exist, create a new list with the value
      expansions[key] = [value]
  # Calculate the average for each key and store it in the dictionary
  for key, value_list in expansions.items():
    expansions[key] = sum(value_list) / len(value_list)
  expansions = tuple(expansions.items())
  depth, avg_expansions = zip(*expansions)

  # save shorted path per moves scrambled
  shortest_paths = {}
  for key, value in moves_scrambled_vs_shortest_path:
    if key in shortest_paths:
      # If the key already exists in the dictionary, add the value to the list
      shortest_paths[key].append(value)
    else:
      # If the key doesn't exist, create a new list with the value
      shortest_paths[key] = [value]
  # Calculate the average for each key and store it in the dictionary
  for key, value_list in shortest_paths.items():
    shortest_paths[key] = sum(value_list) / len(value_list)
  shortest_paths = tuple(shortest_paths.items())
  scrambled_moves, cost = zip(*shortest_paths)

  # Graph Depth vs Average Expansions
  plt.subplot(1, 2, 1)
  plt.bar(range(len(depth)), avg_expansions)
  x_labels = [category for category, value in expansions]
  plt.xticks(range(len(depth)), x_labels)
  plt.xlabel('Depth')
  plt.ylabel('Average Expansions')
  plt.title('Depth vs Average Expansions')

  # Graph Scrambled Moves vs Length of Shortest Path
  plt.subplot(1, 2, 2)
  plt.bar(range(len(scrambled_moves)), cost)
  x_labels = [category for category, value in shortest_paths]
  plt.xticks(range(len(scrambled_moves)), x_labels)
  plt.xlabel('Scrambled Moves')
  plt.ylabel('Length of Shortest Path')
  plt.title('Scrambled Moves vs Length of Shortest Path')

  plt.tight_layout()
  # Display the graph
  plt.show()



if __name__ == '__main__':
  main()