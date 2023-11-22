"""
This defines the objects needed to represent the pieces on a megaminx cube puzzle
"""

class MegaminxPiece:
  def __init__(self, color, piece_type):
    self.color = color
    self.previous_color = color
    self.piece_type = piece_type


  def get_color(self):
    return self.color
  

  def get_piece_type(self):
    return self.piece_type


class MiddleEdgePiece(MegaminxPiece):
  def __init__(self, color, center, left=None, right=None, opposite=None):
    super().__init__(color, piece_type="MiddleEdgePiece")
    # each piece is oriented with the "bottom" connected to the center.
    # "left" is the left neighbor of the middle edge piece with respect to
    # the lower/south piece being the center piece for that face

    # left and right will always be CornerEdgePieces
    self.left = left
    self.right = right
    # this will always be a CenterFacePiece
    self.center = center
    # this will always be another MiddleEdgePiece
    self.opposite = opposite


  def get_face_color(self):
    return self.center.get_color()
  

  def create_opposite_connection(self, center_1, center_2):
    center_1.opposite = center_2
    center_2.opposite = center_1
  

class CenterFacePiece(MegaminxPiece):
  def __init__(self, color, connected_faces, center_edge_pieces=[]):
    super().__init__(color, piece_type="CenterFacePiece")
    # these will always be MiddleEdgePieces
    # 1 is top of upright star, labels increase clockwise
    self.center_edge_pieces = center_edge_pieces
    # the opposite CenterFacePiece i.e. white is always across from white
    # this is an array of center face pieces that are connected to this face
    self.connected_faces = connected_faces


  def insert_center_edge(self, center_edge):
    if len(self.center_edge_pieces) >= 5:
      print('There are already max connections on this center face piece')
    else:
      self.center_edge_pieces.append(center_edge)


class CornerEdgePiece(MegaminxPiece):
  def __init__(self, color, left, right, left_opposite=None, right_opposite=None):
    super().__init__(color, piece_type="CornerEdgePiece")
    # left and right will always be MiddleEdgePieces
    self.left = left
    self.right = right
    # these will always be other CornerEdgePieces
    self.left_opposite = left_opposite
    self.right_opposite = right_opposite
  
  
  def get_face_color(self):
    return self.center.get_color()

