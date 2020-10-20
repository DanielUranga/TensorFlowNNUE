import chess
from enum import Enum
from enum import IntFlag
import tensorflow as tf

SQUARE_NB = 64

class PieceSquare(IntFlag):
    NONE = 0,
    W_PAWN = 1,
    B_PAWN = 1 * SQUARE_NB + 1
    W_KNIGHT = 2 * SQUARE_NB + 1
    B_KNIGHT = 3 * SQUARE_NB + 1
    W_BISHOP = 4 * SQUARE_NB + 1
    B_BISHOP = 5 * SQUARE_NB + 1
    W_ROOK = 6 * SQUARE_NB + 1
    B_ROOK = 7 * SQUARE_NB + 1
    W_QUEEN = 8 * SQUARE_NB + 1
    B_QUEEN = 9 * SQUARE_NB + 1
    W_KING = 10 * SQUARE_NB + 1
    END = W_KING  # pieces without kings (pawns included)
    B_KING = 11 * SQUARE_NB + 1
    END2 = 12 * SQUARE_NB + 1

    @staticmethod
    def from_piece(p: chess.Piece, is_white_pov: bool):
        return {
            chess.WHITE: {
                chess.PAWN: PieceSquare.W_PAWN,
                chess.KNIGHT: PieceSquare.W_KNIGHT,
                chess.BISHOP: PieceSquare.W_BISHOP,
                chess.ROOK: PieceSquare.W_ROOK,
                chess.QUEEN: PieceSquare.W_QUEEN,
                chess.KING: PieceSquare.W_KING
            },
            chess.BLACK: {
                chess.PAWN: PieceSquare.B_PAWN,
                chess.KNIGHT: PieceSquare.B_KNIGHT,
                chess.BISHOP: PieceSquare.B_BISHOP,
                chess.ROOK: PieceSquare.B_ROOK,
                chess.QUEEN: PieceSquare.B_QUEEN,
                chess.KING: PieceSquare.B_KING
            }
        }[p.color == is_white_pov][p.piece_type]

def orient(is_white_pov: bool, sq: int):
    # Use this one for "flip" instead of "rotate"
    # return (chess.A8 * (not is_white_pov)) ^ sq
    return (63 * (not is_white_pov)) ^ sq

def make_halfkp_index(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
    return orient(is_white_pov, sq) + PieceSquare.from_piece(p, is_white_pov) + PieceSquare.END * king_sq

# Returns SparseTensors
def get_halfkp_indeces(board: chess.Board):
    result = []
    for turn in [board.turn, not board.turn]:
        indices = []
        values = []
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            indices.append([0, make_halfkp_index(turn, orient(turn, board.king(turn)), sq, p)])
            values.append(1)
        result.append(tf.sparse.reorder(tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[1, 41024])))
    return result
