"""Decoders (C4): turn per-frame emissions into one steady label per graded frame."""

from rondo.decoder.viterbi import ViterbiParams, decode

__all__ = ["ViterbiParams", "decode"]
