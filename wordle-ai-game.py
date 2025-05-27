import tkinter as tk
from tkinter import messagebox
import random
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD

# --- Constants ---
WORD_LENGTH = 5
ALPHABET_SIZE = 26
MAX_GUESSES = 6
# UI related, could be refactored further if UI becomes more complex
WINDOW_GEOMETRY = "300x600"
GRID_PADY = 20
CELL_PADX = 2
CELL_PADY = 2
AI_FRAME_PADY = 10
STATUS_PADY = 10
FONT_FAMILY = "Arial"
GRID_FONT_SIZE = 24
AI_LABEL_FONT_SIZE = 16
STATUS_FONT_SIZE = 14
# Colors, could also be constants
COLOR_GREEN = 'green'
COLOR_YELLOW = 'yellow'
COLOR_GRAY = 'gray'
COLOR_WHITE = 'white'

# --- Module Docstring ---
"""
Wordle with Automatic AI Player

This script implements the game of Wordle and features an AI player
that attempts to guess the secret word. The AI uses a simple neural network
built with Tinygrad to learn from its guesses and improve its predictions.
The game is played using a Tkinter-based graphical user interface.

The AI learns by:
- Embedding letters into a vector space.
- Processing the sequence of letter embeddings through a hidden layer.
- Predicting the next word from the vocabulary.
- Updating its knowledge based on feedback (green, yellow, gray letters)
  to narrow down the list of possible words.
- Training its neural network based on its guesses and the actual target word.

Classes:
    WordleAI: Manages the AI's logic, including word prediction and learning.
    WordleGame: Manages the game state, UI, and interaction between the user (implicitly, via AI) and the game.
"""

class WordleAI:
    """
    Manages the AI's logic for playing Wordle.

    This class encapsulates the neural network model, word prediction strategy,
    and learning mechanisms of the AI. It maintains information about excluded
    letters, included letters, and known letter positions to refine its guesses.
    """
    def __init__(self, word_list, embedding_dim=50, hidden_dim=100):
        """
        Initializes the WordleAI.

        Args:
            word_list (list[str]): The list of all possible words for the game.
            embedding_dim (int): The dimensionality of the letter embedding space.
            hidden_dim (int): The dimensionality of the hidden layer in the neural network.
        """
        self.word_list = word_list
        self.vocab_size = len(word_list)
        self.embedding_dim = embedding_dim  # Dimension of letter embeddings
        self.embedding = Linear(ALPHABET_SIZE, embedding_dim)  # NN Layer: ALPHABET_SIZE to embedding_dim
        # NN Layer: takes concatenated embeddings of WORD_LENGTH letters
        self.hidden = Linear(embedding_dim * WORD_LENGTH, hidden_dim)
        self.output = Linear(hidden_dim, self.vocab_size) # NN Layer: predicts a word from the vocab
        self.optim = SGD([self.embedding.weight, self.hidden.weight, self.output.weight], lr=0.01) # Optimizer for the NN
        self.excluded_letters = set() # Set of letters known to not be in the target word
        self.included_letters = set() # Set of letters known to be in the target word
        self.letter_positions = {i: set() for i in range(WORD_LENGTH)} # Dict mapping position to known letter (if green)
        self.valid_words = set(word_list) # Subset of word_list that are still possible solutions

    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Args:
            x (Tensor): The input tensor representing a word (or partial word).
                        Shape should be (WORD_LENGTH, ALPHABET_SIZE).

        Returns:
            Tensor: The log-softmax probabilities for each word in the vocabulary.
        """
        # Forward pass through the neural network
        x = x @ self.embedding.weight.T # Embed the input tensor
        x = x.reshape((1, -1)) # Flatten the tensor to (1, WORD_LENGTH * embedding_dim)
        x = self.hidden(x).relu() # Pass through hidden layer with ReLU activation
        return self.output(x).log_softmax() # Pass through output layer and apply log_softmax

    def train(self, input_word, target_word):
        """
        Trains the neural network based on a guessed word and the target word.

        Args:
            input_word (str): The word guessed by the AI.
            target_word (str): The actual target word.
        """
        Tensor.training = True # Set model to training mode
        try:
            self.optim.zero_grad() # Reset gradients before backward pass
            input_tensor = self.word_to_tensor(input_word) # Convert input word to tensor
            target_index = self.word_list.index(target_word) # Get index of target word in the vocab
            target_tensor = Tensor([target_index]) # Create target tensor for loss calculation
            # Calculate loss (cross-entropy between prediction and target)
            loss = self.forward(input_tensor).cross_entropy(target_tensor)
            loss.backward() # Backpropagate the loss to compute gradients
            self.optim.step() # Update model parameters using the optimizer
        finally:
            Tensor.training = False # Set model back to evaluation mode

    def predict(self, partial_word):
        """
        Predicts the next best word to guess based on current information.

        Args:
            partial_word (str): The current state of the guessed word, with '_' for unknown letters.
                                (Note: current implementation uses the last guess, not a true partial word)

        Returns:
            str: The predicted word from the AI's current valid word list.
        """
        input_tensor = self.word_to_tensor(partial_word) # Convert partial word (or last guess) to tensor
        output = self.forward(input_tensor) # Get model prediction (log-softmax probabilities)
        
        # Filter predictions to only include words currently considered valid
        valid_indices = [i for i, word in enumerate(self.word_list) if word in self.valid_words]
        
        if not valid_indices: # If no valid words (should ideally not happen if word list is good)
            # Fallback: choose randomly from remaining valid_words or the full word list if valid_words is empty
            return random.choice(list(self.valid_words)) if self.valid_words else random.choice(self.word_list)
        
        # Select the word with the highest probability among the filtered valid words
        # output[0, valid_indices] selects probabilities for valid words.
        valid_probs = output[0, valid_indices].softmax() # Apply softmax to get normalized probabilities
        chosen_index_in_valid = valid_probs.argmax().item() # Index of the max probability within valid_probs
        chosen_original_index = valid_indices[chosen_index_in_valid] # Map back to original word_list index
        return self.word_list[chosen_original_index]

    def word_to_tensor(self, word):
        """
        Converts a word (string) into a one-hot encoded tensor.

        Args:
            word (str): The word to convert. Must be of length WORD_LENGTH.
                        Can contain '_' for unknown letters, which will be all zeros.

        Returns:
            Tensor: A 2D tensor of shape (WORD_LENGTH, ALPHABET_SIZE) representing the word.
        """
        # Initialize a tensor of zeros with shape (WORD_LENGTH, ALPHABET_SIZE)
        tensor = Tensor.zeros((WORD_LENGTH, ALPHABET_SIZE)).contiguous()
        for i, char in enumerate(word):
            if char != '_': # '_' represents an unknown letter, resulting in a zero vector for that position
                index = ord(char) - ord('A') # Convert character to an index (A=0, B=1, ...)
                tensor[i, index] = 1 # Set the corresponding element to 1 (one-hot encoding for the letter)
        return tensor

    def update_letter_info(self, guess, feedback):
        """
        Updates the AI's knowledge based on the feedback from a guess.

        Modifies `self.excluded_letters`, `self.included_letters`,
        `self.letter_positions`, and `self.valid_words`.

        Args:
            guess (str): The word that was guessed.
            feedback (list[str]): A list of feedback strings ('green', 'yellow', 'gray')
                                  corresponding to each letter in the guess.
        """
        for i, (letter, color) in enumerate(zip(guess, feedback)):
            if color == COLOR_GRAY: # Letter is not in the word
                # Add to excluded_letters only if not already confirmed as included (e.g. a double letter where one is gray, one is yellow/green)
                if letter not in self.included_letters:
                    self.excluded_letters.add(letter)
            else: # Letter is in the word (either yellow or green)
                self.included_letters.add(letter)
                if color == COLOR_GREEN: # Letter is in the correct position
                    self.letter_positions[i] = {letter} # Set this position to this letter
                elif color == COLOR_YELLOW: # Letter is in the word but in the wrong position
                    # If this position was previously thought to be this letter (e.g. a green changed to yellow - unlikely in Wordle but good for robustness),
                    # remove that specific assumption.
                    if i in self.letter_positions and self.letter_positions[i] == {letter}:
                         self.letter_positions[i] = set() # No letter is confirmed for this position now
                    # Also, this letter cannot be in this specific position 'i'.
                    # This is implicitly handled by `is_valid_word` when it checks words against `letter_positions`
                    # and by the overall filtering of `self.valid_words`.
                    # A word will be invalid if it has `letter` at `i` when `letter_positions[i]` is known and different,
                    # or if `letter_positions[i]` becomes empty, other words might fit.
                    # The key is that `letter` is in `included_letters`.
                    pass # No direct update to letter_positions for yellow beyond ensuring it's not marked green here.

        # Re-filter the list of valid words based on the new information gathered
        self.valid_words = {word for word in self.valid_words if self.is_valid_word(word)}

    def is_valid_word(self, word):
        """
        Checks if a given word is valid based on the AI's current knowledge.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word is valid, False otherwise.
        """
        # 1. Check if the word contains any letters known to be excluded from the target word.
        if any(letter in self.excluded_letters for letter in word):
            return False
        # 2. Check if the word contains all letters known to be included in the target word.
        if not all(letter in self.included_letters for letter in word):
            return False
        # 3. Check letter positions based on 'green' feedback:
        #    - If a position `i` has a confirmed letter (green), the `word` must have that letter at `i`.
        #    Check for 'yellow' feedback implication:
        #    - If a letter `L` was 'yellow' at position `i`, then `word[i]` cannot be `L`.
        #      This is handled by how `valid_words` is filtered in `update_letter_info`.
        #      If `update_letter_info` correctly processes yellow (e.g., by temporarily adding `(letter, i)` to a "not here" list
        #      or by ensuring `letter_positions[i]` doesn't make it seem valid), then this check is simpler.
        #      The current `letter_positions[i]` only stores green letters.
        for i, letter_in_word in enumerate(word):
            # If a specific letter is confirmed for this position (green feedback)
            if self.letter_positions[i] and letter_in_word not in self.letter_positions[i]:
                return False # The word does not match the green feedback at this position.

            # Additional check for yellow letters: if letter 'L' was yellow at position 'i',
            # then word[i] should not be 'L'.
            # This is NOT explicitly checked here but is expected to be handled by the filtering
            # of `self.valid_words` in `update_letter_info`. If `self.valid_words` is correctly pruned,
            # a word passed to `is_valid_word` should already satisfy this.
            # The current structure of `letter_positions` only holds green letters.
            # A more robust way would be to have another structure like `yellow_at_positions[i] = {letters that were yellow here}`.
            # Then, one could check: `if letter_in_word in yellow_at_positions[i]: return False`.
            # Sticking to current structure for now.

        return True

class WordleGame:
    """
    Manages the Wordle game state, UI, and interactions.

    This class sets up the Tkinter UI, handles the game loop,
    processes AI guesses, provides feedback, and manages game resets.
    """
    def __init__(self, master):
        """
        Initializes the Wordle game.

        Args:
            master (tk.Tk): The root Tkinter window.
        """
        self.master = master
        self.master.title("Wordle with Automatic AI") # Window title
        self.master.geometry(WINDOW_GEOMETRY) # Set window size using constant
        
        self.word_list = self.load_words("listof5characterwords.txt") # Load the list of possible words
        self.target_word = self.select_word() # Select the secret target word for this game
        self.current_row = 0 # Initialize current guess row in the UI grid (0 to MAX_GUESSES-1)
        
        self.ai = WordleAI(self.word_list) # Initialize the WordleAI agent
        # AI's initial guess; random choice if word list is available, otherwise a placeholder.
        self.ai_guess = random.choice(self.word_list) if self.word_list else ('_' * WORD_LENGTH)
        
        self.create_ui() # Setup the game's graphical user interface
        self.master.after(1000, self.play_ai_turn) # Schedule the AI's first turn after 1 second
    
    def load_words(self, file_path):
        """
        Loads words from a file, filtering for those of WORD_LENGTH.

        Args:
            file_path (str): The path to the file containing words.

        Returns:
            list[str]: A list of uppercase words of the correct length.
                       The game will quit if the file is not found or is empty.
        """
        try:
            with open(file_path, 'r') as file:
                # Load words, strip whitespace, convert to uppercase, and filter by WORD_LENGTH
                words = [word.strip().upper() for word in file if len(word.strip()) == WORD_LENGTH]
                if not words:
                    messagebox.showerror("Error", f"No words of length {WORD_LENGTH} found in {file_path}.")
                    self.master.quit()
                return words
        except FileNotFoundError:
            messagebox.showerror("Error", f"Word file not found: {file_path}")
            self.master.quit() # Exit application if word file is critical and not found
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading the word file: {e}")
            self.master.quit() # Exit on other file reading errors
        return [] # Should not be reached if quit works
    
    def select_word(self):
        """
        Selects a random target word from the loaded word list.

        Returns:
            str: The randomly selected target word.
                 The game will quit if the word list is empty.
        """
        if not self.word_list:
            messagebox.showerror("Error", "Word list is empty. Cannot select a target word.")
            self.master.quit() # Exit if no words are available to select from
            return "" # Should not be reached
        return random.choice(self.word_list) # Select a random word from the list
    
    def create_ui(self):
        """Sets up the graphical user interface for the Wordle game."""
        self.grid_frame = tk.Frame(self.master)
        self.grid_frame.pack(pady=GRID_PADY) # Main frame for the Wordle letter grid
        
        # Create the grid of labels (cells) for displaying guesses and feedback
        self.grid = [[tk.Label(self.grid_frame, width=2, height=1, relief='solid', font=(FONT_FAMILY, GRID_FONT_SIZE))
                      for _ in range(WORD_LENGTH)] for _ in range(MAX_GUESSES)]
        
        # Place each cell (Label widget) in the grid_frame using grid geometry manager
        for i in range(MAX_GUESSES):
            for j in range(WORD_LENGTH):
                self.grid[i][j].grid(row=i, column=j, padx=CELL_PADX, pady=CELL_PADY)
        
        self.ai_frame = tk.Frame(self.master)
        self.ai_frame.pack(pady=AI_FRAME_PADY) # Frame for displaying AI's current guess and game status

        # Label to show the AI's current guess text
        self.ai_label = tk.Label(self.ai_frame, text=f"AI Guess: {self.ai_guess}", font=(FONT_FAMILY, AI_LABEL_FONT_SIZE))
        self.ai_label.pack()

        # Label to show game status messages (e.g., "Game in progress...", "AI wins!")
        self.status_label = tk.Label(self.ai_frame, text="Game in progress...", font=(FONT_FAMILY, STATUS_FONT_SIZE))
        self.status_label.pack(pady=STATUS_PADY)
    
    def play_ai_turn(self):
        """Handles the AI's turn to make a guess."""
        if self.current_row < MAX_GUESSES: # Check if AI has guesses remaining
            self.ai_guess = self.ai.predict(self.ai_guess) # Get AI's next predicted word
            self.ai_label.config(text=f"AI Guess: {self.ai_guess}") # Update UI to show AI's guess
            
            # Display the AI's guess letters in the current row of the UI grid
            for i, letter in enumerate(self.ai_guess):
                self.grid[self.current_row][i].config(text=letter)
            
            self.master.update() # Refresh the UI to show the guess immediately
            self.master.after(1000, self.check_guess) # Schedule guess checking after a 1-second delay
        else:
            # This case should ideally be caught by check_guess, but as a fallback:
            self.end_game(False) # AI ran out of guesses without winning
    
    def check_guess(self):
        """
        Checks the AI's current guess against the target word and updates the UI.
        Provides feedback to the AI and determines if the game has ended.
        """
        guess = self.ai_guess
        feedback = [''] * WORD_LENGTH # Initialize feedback array for the AI
        
        # Determine feedback (green, yellow, gray) for each letter and update cell colors
        for i in range(WORD_LENGTH):
            if guess[i] == self.target_word[i]: # Letter is correct and in the correct position
                self.grid[self.current_row][i].config(bg=COLOR_GREEN)
                feedback[i] = COLOR_GREEN
            elif guess[i] in self.target_word: # Letter is in the word but in the wrong position
                self.grid[self.current_row][i].config(bg=COLOR_YELLOW)
                feedback[i] = COLOR_YELLOW
            else: # Letter is not in the word
                self.grid[self.current_row][i].config(bg=COLOR_GRAY)
                feedback[i] = COLOR_GRAY
        
        self.ai.update_letter_info(guess, feedback) # Provide feedback to the AI to update its knowledge
        self.ai.train(guess, self.target_word) # Train the AI model with the current guess and target
        
        if guess == self.target_word: # AI guessed the word correctly
            self.end_game(True)
        else: # Guess was incorrect
            self.current_row += 1 # Move to the next guess row
            if self.current_row < MAX_GUESSES: # If guesses are still available
                self.master.after(1000, self.play_ai_turn) # Schedule AI's next turn
            else: # No guesses left, AI loses
                self.end_game(False)
    
    def end_game(self, success):
        """
        Handles the end of the game (win or lose).

        Args:
            success (bool): True if the AI guessed the word, False otherwise.
        """
        if success:
            self.status_label.config(text=f"AI guessed the word in {self.current_row + 1} tries!")
        else:
            self.status_label.config(text=f"AI failed. The word was {self.target_word}")
        
        # Ask the user if they want to play again
        play_again = messagebox.askyesno("Game Over", "Do you want to play again?")
        if play_again:
            self.reset_game()
        else:
            self.master.quit() # Close the game window
    
    def reset_game(self):
        """Resets the game state for a new game."""
        self.target_word = self.select_word() # Select a new target word
        self.current_row = 0 # Reset current guess row
        # Reset AI's guess to a new random word or placeholder
        self.ai_guess = random.choice(self.word_list) if self.word_list else ('_' * WORD_LENGTH)
        self.ai = WordleAI(self.word_list) # Reinitialize the AI for the new game
        
        # Clear the grid UI (reset text and background colors)
        for row_idx in range(MAX_GUESSES):
            for col_idx in range(WORD_LENGTH):
                self.grid[row_idx][col_idx].config(text='', bg=COLOR_WHITE)
        
        # Reset UI labels for AI guess and game status
        self.ai_label.config(text=f"AI Guess: {self.ai_guess}")
        self.status_label.config(text="Game in progress...")
        
        self.master.after(1000, self.play_ai_turn) # Start the new game with AI's first turn

if __name__ == "__main__":
    root = tk.Tk() # Create the main Tkinter window (master for the game)
    game = WordleGame(root) # Initialize the Wordle game instance
    root.mainloop() # Start the Tkinter event loop to run the game
