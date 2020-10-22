#!/usr/bin/env python3

import numpy as np
import keras
import chess
import chess.pgn
import random

from halfkp import get_halfkp_indeces

GAMES_PER_EPOCH = 1_000

def gen(pgn_file_path):
  with open(pgn_file_path) as pgn:

    # Start from random position in the file
    pgn.seek(0, 2)
    pgn.seek(random.randint(0, pgn.tell()))
    chess.pgn.read_headers(pgn)

    while True:
      game = chess.pgn.read_game(pgn)

      if not game:
        pgn.seek(0)
        continue

      '''
      result_header = game.headers['Result']
      game_value_for_white = 0
      if result_header == '*':
        continue
      elif result_header == '1-0':
        game_value_for_white = 1
      elif result_header == '0-1':
        game_value_for_white = -1
      else:
        game_value_for_white = 0
      '''

      board = game.board()
      for node in game.mainline():
        board.push(node.move)
        eval = node.eval()
        if not eval:
          break
        eval = eval.pov(not board.turn).score()
        if not eval:
          break
        X = get_halfkp_indeces(board)
        # y = game_value_for_white if board.turn == chess.WHITE else -game_value_for_white
        # y = eval if board.turn == chess.WHITE else -eval
        yield (X[0], X[1]), eval / 64
