#!/usr/bin/env python3

import numpy as np
import keras
import chess

GAMES_PER_EPOCH = 1_000

class DataGenerator(keras.utils.Sequence):
  def __init__(self, pgn_file_path, batch_size=32):
    'Initialization'
    self.pgn_file_path = pgn_file_path
    self.batch_size = batch_size
    self.on_epoch_end()
    self.pgn_file_offset = 0 # TODO: Randomize

  def on_epoch_end(self):
    'on_epoch_end'

    self.Xs = []
    self.ys = []

    with open(self.pgn_file_path) as pgn:
      pgn.seek(self.pgn_file_offset)
      game_count = 0

      while game_count < GAMES_PER_EPOCH:
          game = chess.pgn.read_game(pgn)
          if not game:
            self.pgn_file_offset = 0
            break

          board = game.board()
          for move in game.mainline_moves():
            board.push(move)

          game_count = game_count + 1

      self.pgn_file_offset = pgn.tell()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.Xs) / self.batch_size))

  def __getitem__(self, batch_index):
    'Generate one batch of data'
    Xs = self.Xs[batch_index:batch_index+self.batch_size]
    ys = self.ys[batch_index:batch_index+self.batch_size]
    return Xs, ys