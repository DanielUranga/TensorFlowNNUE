#!/usr/bin/env python3

import numpy as np
import keras
import chess
import chess.pgn

from halfkp import get_halfkp_indeces

GAMES_PER_EPOCH = 1_000

def gen(pgn_file_path):
  with open(pgn_file_path) as pgn:
    while True:
      game = chess.pgn.read_game(pgn)
      if not game:
        break

      result_header = game.headers['Result']
      result_vals = [0, 0] # Result "values" for black and white
      if result_header == '*':
        continue
      elif result_header == '1-0':
        result_vals[0] = 1
        result_vals[1] = 0
      elif result_header == '0-1':
        result_vals[0] = 0
        result_vals[1] = 1
      else:
        result_vals[0] = 0
        result_vals[1] = 0

      board = game.board()
      for move in game.mainline_moves():
        board.push(move)
        X = get_halfkp_indeces(board)
        turn_idx = 0 if board.turn == chess.WHITE else 1
        y = result_vals[turn_idx]
        yield (X[0], X[1]), y
