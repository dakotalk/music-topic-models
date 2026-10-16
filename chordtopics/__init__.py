"""Topic models of chord progressions in popular music.

Modules
-------
chords        shared chord-spelling utilities (pitch classes, transposition, simplification)
billboard     parser for the McGill Billboard SALAMI chord annotations
chordonomicon loader for the Chordonomicon chord-progression dataset
tokenize      bargram tokenizer, collocation detection, vocabulary thresholding
model         LDA training, held-out coherence, seed stability
evaluate      downstream tasks with baselines (section labels, year prediction)
"""
