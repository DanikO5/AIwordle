# Wordle AI Game

A Python-based Wordle game where an AI player attempts to guess a hidden 5-letter word. The game provides visual feedback (green, yellow, gray squares) which the AI uses to improve its subsequent guesses.

## How it Works

The game is built using `tkinter` for the graphical user interface. The AI opponent uses a simple neural network implemented with the `tinygrad` library. It learns from the feedback of each guess to narrow down the possibilities and predict the target word.

## Dependencies

*   Python 3.x
*   `tkinter` (typically included with standard Python installations)
*   `tinygrad`

## Installation

1.  Ensure you have Python 3 installed.
2.  Install `tinygrad`:
    ```bash
    pip install tinygrad
    ```

## How to Run

1.  Clone this repository or download the files.
2.  Navigate to the directory containing the files.
3.  Run the game using the following command:
    ```bash
    python wordle-ai-game.py
    ```
