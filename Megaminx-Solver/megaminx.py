"""
Project 1 for CS463G 9/10/23
"""
import megaminx_pieces as mp
import random
from collections import deque, defaultdict
MAX_FACE_CONNECTIONS = 5
MAX_CORNER_PIECES_PER_FACE = 5
MAX_CENTER_EDGE_PIECES_PER_FACE = 5


class Megaminx():
  def __init__(self):
    # WH=WHITE, YE=YELLOW, DB=DARK BLUE, RE=RED
    # DG=DARK GREEN, PR=PURPLE, GR=GREY, BE=BEIGE
    # LG=LIGHT GREEN, OR=ORANGE, LB=LIGHT BLUE, PI=PINK
    # Colors modeled after this 3d model: https://www.grubiks.com/puzzles/megaminx-3x3x3/
    # connections were also created by this, based on the attached colors in the model
    self.colors = ["WH", "YE", "DB", "RE", 
                   "DG", "PR", "GR", "BE", 
                   "LG", "OR", "LB", "PI"]

    # key is face, values are neighboring faces
    self.face_connections = {
      "WH": ["YE", "DB", "RE", "DG", "PR"],
      "YE": ["WH", "DB", "LG", "OR", "PR"],
      "DB": ["WH", "YE", "LG", "PI", "RE"],
      "RE": ["WH", "DB", "PI", "BE", "DG"],
      "DG": ["WH", "RE", "BE", "LB", "PR"],
      "PR": ["WH", "DG", "LB", "OR", "YE"],
      "GR": ["LB", "BE", "PI", "LG", "OR"],
      "BE": ["GR", "LB", "DG", "RE", "PI"],
      "LG": ["DB", "YE", "OR", "GR", "PI"],
      "OR": ["YE", "PR", "LB", "GR", "LG"],
      "LB": ["PR", "DG", "BE", "GR", "OR"],
      "PI": ["GR", "BE", "RE", "DB", "LG"]}
    
    # create and save faces
    self.faces = []                                                                                                                          
    for color in self.face_connections:
      # create the face cneter piece
      face_center = mp.CenterFacePiece(color, connected_faces=self.face_connections[color], center_edge_pieces=[])

      # create and link center edge pieces to face center piece
      for i in range(MAX_CENTER_EDGE_PIECES_PER_FACE):
        mid_edge_piece = mp.MiddleEdgePiece(color, face_center)
        face_center.insert_center_edge(mid_edge_piece)
      
      # create and link corner pieces to center edge pieces
      for i in range(MAX_CORNER_PIECES_PER_FACE):
        if i + 1 < MAX_CORNER_PIECES_PER_FACE:
          corner_piece = mp.CornerEdgePiece(color, left=face_center.center_edge_pieces[i], right=face_center.center_edge_pieces[i+1])
          face_center.center_edge_pieces[i].right = corner_piece
          face_center.center_edge_pieces[i+1].left = corner_piece
      corner_piece = mp.CornerEdgePiece(color, left=face_center.center_edge_pieces[MAX_CORNER_PIECES_PER_FACE-1], right=face_center.center_edge_pieces[0])
      face_center.center_edge_pieces[MAX_CORNER_PIECES_PER_FACE-1].right = corner_piece
      face_center.center_edge_pieces[0].left = corner_piece

      self.faces.append(face_center)
    # link all faces together
    self.create_megaminx()

    self.face_distances = self.compute_all_distances()

  def get_face_by_color(self, color):
    for face in self.faces:
      if face.color == color:
        return face


  """
  GPT 3.5 Generated 9/14/23 7pm 
  Prompt: self.face_connections = {
      "WH": ["YE", "DB", "RE", "DG", "PR"],
      "YE": ["WH", "DB", "LG", "OR", "PR"],
      "DB": ["WH", "YE", "LG", "PI", "RE"],
      "RE": ["WH", "DB", "PI", "BE", "DG"],
      "DG": ["WH", "RE", "BE", "LB", "PR"],
      "PR": ["WH", "DG", "LB", "OR", "YE"],
      "GR": ["LB", "BE", "PI", "LG", "OR"],
      "BE": ["GR", "LB", "DG", "RE", "PI"],
      "LG": ["DB", "YE", "OR", "GR", "PI"],
      "OR": ["YE", "PR", "LB", "GR", "LG"],
      "LB": ["PR", "DG", "BE", "GR", "OR"],
      "PI": ["GR", "BE", "RE", "DB", "LG"]}

  I have a list of connections. Give me a function to take any given two colors and return the distance between them
  """
  def shortest_distance_between_colors(self, start_color, end_color):
    if start_color == end_color:
        return 0  # Same color, no distance

    visited = set()
    queue = deque([(start_color, 0)])

    while queue:
        current_color, distance = queue.popleft()
        visited.add(current_color)

        # Check if we've reached the end_color
        if current_color == end_color:
            return distance

        # Add unvisited neighboring colors to the queue
        for neighbor_color in self.face_connections[current_color]:
            if neighbor_color not in visited:
                queue.append((neighbor_color, distance + 1))

    return float('inf') 
  
  """
  GPT 3.5 9/14/23 7:08pm
  Now take this functions, compare every color with one another and create a dictionary that holds each color and the distance to every other color
  (function referenced above)
  """
  def compute_all_distances(self):
    colors = list(self.face_connections.keys())
    distance_dict = defaultdict(dict)

    for color1 in colors:
      for color2 in colors:
        distance = self.shortest_distance_between_colors(color1, color2)
        distance_dict[color1][color2] = distance

    return dict(distance_dict)
  

  def insert_edge_link(self, center_1, center_2):
    """
    Link two edges together
    This process creates assigns "pointers" to every neighboring piece between
    two edges for both middle edge pieces and corner edge pieces
    """
    if center_1.opposite != None:
      print('center 1 already has an opposite')
      return
    center_1.opposite = center_2
    if center_2.opposite != None:
      print('center 2 already has an opposite')
      return
    center_2.opposite = center_1
    if center_1.left.right_opposite != None:
      print('center 1 left already has an right opposite')
      return
    center_1.left.right_opposite = center_2.right
    if center_2.right.left_opposite != None:
      print('center 2 right already has an left opposite')
      return
    center_2.right.left_opposite = center_1.left
    if center_1.right.left_opposite != None:
      print('center 1 right already has an left opposite')
      return
    center_1.right.left_opposite = center_2.left
    if center_2.left.right_opposite != None:
      print('center 2 left already has an right opposite')
      return
    center_2.left.right_opposite = center_1.right


  def connect_face(self, face_1_color, face_1_index, face_2_color, face_2_index):
    if face_1_index >= MAX_CENTER_EDGE_PIECES_PER_FACE:
      print('index out of range for face 1 face connection index')
      return
    if face_2_index >= MAX_CENTER_EDGE_PIECES_PER_FACE:
      print('index out of range for face 2 face connection index')
      return
    face_1 = self.get_face_by_color(face_1_color)
    face_2 = self.get_face_by_color(face_2_color)
    face_1_center_edge = face_1.center_edge_pieces[face_1_index]
    face_2_center_edge = face_2.center_edge_pieces[face_2_index]
    # link center edge 1 and 2 both ways
    self.insert_edge_link(face_1_center_edge, face_2_center_edge)


  def create_megaminx(self):
    self.connect_face("WH", 0, "DG", 2)
    self.connect_face("WH", 1, "PR", 2)
    self.connect_face("WH", 2, "YE", 2)
    self.connect_face("WH", 3, "DB", 2)
    self.connect_face("WH", 4, "RE", 2)

    # connect all faces adjacent to white to eachother (around white)
    self.connect_face("DG", 1, "PR", 3)
    self.connect_face("PR", 1, "YE", 3)
    self.connect_face("YE", 1, "DB", 3)
    self.connect_face("DB", 1, "RE", 3)
    self.connect_face("RE", 1, "DG", 3)

    # connect all faces adjacent to grey to grey
    self.connect_face("GR", 0, "LG", 2)
    self.connect_face("GR", 1, "OR", 2)
    self.connect_face("GR", 2, "LB", 2)
    self.connect_face("GR", 3, "BE", 2)
    self.connect_face("GR", 4, "PI", 2)

    # connect all faces adjacent to grey to eachother (around grey)
    self.connect_face("LG", 1, "OR", 3)
    self.connect_face("OR", 1, "LB", 3)
    self.connect_face("LB", 1, "BE", 3)
    self.connect_face("BE", 1, "PI", 3)
    self.connect_face("PI", 1, "LG", 3)

    # connect all remianing faces
    self.connect_face("LG", 0, "YE", 0)
    self.connect_face("OR", 4, "YE", 4)
    self.connect_face("OR", 0, "PR", 0)
    self.connect_face("LB", 4, "PR", 4)
    self.connect_face("LB", 0, "DG", 0)
    self.connect_face("BE", 4, "DG", 4)
    self.connect_face("BE", 0, "RE", 0)
    self.connect_face("PI", 4, "RE", 4)
    self.connect_face("PI", 0, "DB", 0)
    self.connect_face("LG", 4, "DB", 4)


  def is_megaminx_connected(self):
    is_connected = True
    for face in self.faces:
      if (len(face.center_edge_pieces) != MAX_CENTER_EDGE_PIECES_PER_FACE):
        print(face.color, ' is missing a face connection')
        is_connected = False
      for center_edge_piece in face.center_edge_pieces:
        if center_edge_piece.opposite.color not in self.face_connections[face.color]:
          is_connected = False
    return is_connected
  

  def get_piece_colors_on_face(self, face_color):
    face = self.get_face_by_color(face_color)
    center_pieces = face.center_edge_pieces
    piece_colors = []
    piece_colors.append(center_pieces[0].color)
    piece_colors.append(center_pieces[0].right.color)
    piece_colors.append(center_pieces[1].color)
    piece_colors.append(center_pieces[1].right.color)
    piece_colors.append(center_pieces[2].color)
    piece_colors.append(center_pieces[2].right.color)
    piece_colors.append(center_pieces[3].color)
    piece_colors.append(center_pieces[3].right.color)
    piece_colors.append(center_pieces[4].color)
    piece_colors.append(center_pieces[4].right.color)
    return piece_colors
  

  def is_solved(self):
    all_piece_colors = {}
    for color in self.colors:
      all_piece_colors[color] = self.get_piece_colors_on_face(color)
    for face_color in all_piece_colors:
      for color in all_piece_colors[face_color]:
        if color != face_color:
          return False
    return True
  

  def rotate_clockwise(self, face_color):
    face = self.get_face_by_color(face_color)
    for i in range(len(face.center_edge_pieces)):
      # save the state of the edges before the rotate
      face.center_edge_pieces[i].left.right_opposite.previous_color = face.center_edge_pieces[i].left.right_opposite.color
      face.center_edge_pieces[i].opposite.previous_color = face.center_edge_pieces[i].opposite.color
      face.center_edge_pieces[i].right.left_opposite.previous_color = face.center_edge_pieces[i].right.left_opposite.color
    for i in range(len(face.center_edge_pieces)):
      next_index = (i + 1) % MAX_CENTER_EDGE_PIECES_PER_FACE
      current_index = i
      # assign "next" edge the current edge's saved colors
      face.center_edge_pieces[next_index].left.right_opposite.color = face.center_edge_pieces[current_index].left.right_opposite.previous_color
      face.center_edge_pieces[next_index].opposite.color = face.center_edge_pieces[current_index].opposite.previous_color
      face.center_edge_pieces[next_index].right.left_opposite.color = face.center_edge_pieces[current_index].right.left_opposite.previous_color

    # rotate the current face's pieces
    last_center_edge = face.center_edge_pieces[-1]
    for i in range(len(face.center_edge_pieces) -1, 0, -1):
      face.center_edge_pieces[i] = face.center_edge_pieces[i-1]
    face.center_edge_pieces[0] = last_center_edge
    


  def rotate_counter_clockwise(self, face_color):
    face = self.get_face_by_color(face_color)
    # i.e. swap 0->4, 4->3
    mapping = {0: 4, 4: 3, 3: 2, 2: 1, 1: 0}
    for i in range(len(face.center_edge_pieces)):
      # save the state of the edges before the rotate
      face.center_edge_pieces[i].left.right_opposite.previous_color = face.center_edge_pieces[i].left.right_opposite.color
      face.center_edge_pieces[i].opposite.previous_color = face.center_edge_pieces[i].opposite.color
      face.center_edge_pieces[i].right.left_opposite.previous_color = face.center_edge_pieces[i].right.left_opposite.color
    for key in mapping:
      # assign "next" edge the current edge's saved colors
      face.center_edge_pieces[mapping[key]].left.right_opposite.color = face.center_edge_pieces[key].left.right_opposite.previous_color
      face.center_edge_pieces[mapping[key]].opposite.color = face.center_edge_pieces[key].opposite.previous_color
      face.center_edge_pieces[mapping[key]].right.left_opposite.color = face.center_edge_pieces[key].right.left_opposite.previous_color

    # rotate the current face's pieces
    first_center_edge = face.center_edge_pieces[0]
    for i in range(len(face.center_edge_pieces) -1):
      face.center_edge_pieces[i] = face.center_edge_pieces[i+1]
    face.center_edge_pieces[-1] = first_center_edge


  def random_move(self, num_moves, direction):
    """
    move the megaminx in 'direction' 'num_moves' amount of times
    a face will randomly be chosen per rotation
    """
    random.seed()
    if direction == 'cw':
      for i in range(num_moves):
        face_color = self.colors[random.randint(0,11)]
        self.rotate_clockwise(face_color)

    elif direction == 'ccw':
      for i in range(num_moves):
        face_color = self.colors[random.randint(0,11)]
        self.rotate_counter_clockwise(face_color)

    else:
      print('invalid direction to rotate')
      return
  

  def print_megaminx(self, num_faces_per_line):
    """
    Print every face layer by layer (i.e. print all "top" pieces of the pentagons, then next two, etc.)
    Source: https://github.com/Noahhhughes/MegaminxSolver/tree/master
      I referenced this repo, specifically the Main.py to determine how to print hexagons side-by-side in text
      Referenced: 9/7/23, Author: Noahhhughes on Github, Title: MegaminxSolver on Github
      I altered this to be able to print a variable amount of faces per line
      i.e. print 4 faces, then under that print another 4, etc.
    """
    for j in range((len(self.faces)//num_faces_per_line)+1):
      for i in range(num_faces_per_line):
        if j*num_faces_per_line+i < len(self.faces):
          face = self.faces[j*num_faces_per_line+i]
          print("      ", face.center_edge_pieces[0].left.color, end='')
          print("           ", end='')
      if j * num_faces_per_line < len(self.faces):
       print()
      for i in range(num_faces_per_line):
        if j*num_faces_per_line+i < len(self.faces):
          face = self.faces[j*num_faces_per_line+i]
          print("  ", face.center_edge_pieces[4].color, "    ", face.center_edge_pieces[0].color, end='')
          print("       ", end='')
      if j * num_faces_per_line < len(self.faces):
       print()
      for i in range(num_faces_per_line):
        if j*num_faces_per_line+i < len(self.faces):
          face = self.faces[j*num_faces_per_line+i]
          print(face.center_edge_pieces[4].left.color, "          ", face.center_edge_pieces[0].right.color, end='')
          print("    ", end='')
      if j * num_faces_per_line < len(self.faces):
       print()
      for i in range(num_faces_per_line):
        if j*num_faces_per_line+i < len(self.faces):
          face = self.faces[j*num_faces_per_line+i]
          print(" ", face.center_edge_pieces[3].color, "      ", face.center_edge_pieces[1].color, end='')
          print("      ", end='')
      if j * num_faces_per_line < len(self.faces):
       print()
      for i in range(num_faces_per_line):
        if j*num_faces_per_line+i < len(self.faces):
          face = self.faces[j*num_faces_per_line+i]
          print("   ", face.center_edge_pieces[3].left.color, face.center_edge_pieces[2].color, face.center_edge_pieces[1].right.color, end='')
          print("        ", end='')
      if j * num_faces_per_line < len(self.faces):
       print()
       print()

      # Print name of face, i.e. face center peice color
      for i in range(num_faces_per_line):
        if j*num_faces_per_line+i < len(self.faces):
          face = self.faces[j*num_faces_per_line+i]
          print("  ", face.color, "side", end='')
          print("          ", end='')
      if j * num_faces_per_line < len(self.faces):
       print()
       print()


def main():
  minx = Megaminx()
  print('Are all faces on megaminx connected: ', minx.is_megaminx_connected())
  

  # # uncomment to rotate a certain face clockwise
  # # reference the megaminx colors to choose
  # minx.rotate_clockwise("WH")

  # # uncomment to rotate a certain face counter_clockwise
  # # reference the megaminx colors to choose
  # minx.rotate_counter_clockwise("WH")

  # # uncomment to test rotating a face 5 times (should be same as beginging)
  # for key in minx.face_connections:
  #   minx.rotate_counter_clockwise(key)
  #   minx.rotate_counter_clockwise(key)
  #   minx.rotate_counter_clockwise(key)
  #   minx.rotate_counter_clockwise(key)
  #   minx.rotate_counter_clockwise(key)

  # # uncomment to make 2 random moves in the clockwise direction
  # minx.random_move(2, 'cw')

  # # uncomment to make 2 random moves in the counter-clockwise direction
  minx.random_move(2, 'ccw')
  print(minx.is_solved())

  # define number of faces you want to print per line
  minx.print_megaminx(4)

  
  
if __name__ == '__main__':
  main()