import tkinter as tk
from tkinter import messagebox
import random
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD

class WordleAI:
    def __init__(self, word_list, embedding_dim=50, hidden_dim=100):
        self.word_list = word_list
        self.vocab_size = len(word_list)
        self.embedding_dim = embedding_dim
        self.embedding = Linear(26, embedding_dim)  # 26 letters
        self.hidden = Linear(embedding_dim * 5, hidden_dim)
        self.output = Linear(hidden_dim, self.vocab_size)
        self.optim = SGD([self.embedding.weight, self.hidden.weight, self.output.weight], lr=0.01)
        self.excluded_letters = set()
        self.included_letters = set()
        self.letter_positions = {i: set() for i in range(5)}
        self.valid_words = set(word_list)

    def forward(self, x):
        x = x @ self.embedding.weight.T
        x = x.reshape((1, -1))
        x = self.hidden(x).relu()
        return self.output(x).log_softmax()

    def train(self, input_word, target_word):
        Tensor.training = True
        try:
            self.optim.zero_grad()
            input_tensor = self.word_to_tensor(input_word)
            target_index = self.word_list.index(target_word)
            target_tensor = Tensor([target_index])
            loss = self.forward(input_tensor).cross_entropy(target_tensor)
            loss.backward()
            self.optim.step()
        finally:
            Tensor.training = False

    def predict(self, partial_word):
        input_tensor = self.word_to_tensor(partial_word)
        output = self.forward(input_tensor)
        
        valid_indices = [i for i, word in enumerate(self.word_list) if word in self.valid_words]
        
        if not valid_indices:
            return random.choice(list(self.valid_words)) if self.valid_words else random.choice(self.word_list)
        
        valid_probs = output[0, valid_indices].softmax()
        chosen_index = valid_indices[valid_probs.argmax().item()]
        return self.word_list[chosen_index]

    def word_to_tensor(self, word):
        tensor = Tensor.zeros((5, 26)).contiguous()
        for i, char in enumerate(word):
            if char != '_':
                index = ord(char) - ord('A')
                tensor[i, index] = 1
        return tensor

    def update_letter_info(self, guess, feedback):
        for i, (letter, color) in enumerate(zip(guess, feedback)):
            if color == 'gray':
                if letter not in self.included_letters:
                    self.excluded_letters.add(letter)
            else:
                self.included_letters.add(letter)
                if color == 'green':
                    self.letter_positions[i] = {letter}
                elif color == 'yellow':
                    if i in self.letter_positions:
                        self.letter_positions[i].discard(letter)
        
        self.valid_words = {word for word in self.valid_words if self.is_valid_word(word)}

    def is_valid_word(self, word):
        if any(letter in self.excluded_letters for letter in word):
            return False
        if not all(letter in word for letter in self.included_letters):
            return False
        for i, letter in enumerate(word):
            if self.letter_positions[i] and letter not in self.letter_positions[i]:
                return False
        return True

class WordleGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Wordle with Automatic AI")
        self.master.geometry("300x600")
        
        self.word_list = self.load_words(r"C:\Users\danik\Desktop\wordle\cleaned_words.txt")
        self.target_word = self.select_word()
        self.current_row = 0
        
        self.ai = WordleAI(self.word_list)
        self.ai_guess = '_____'
        
        self.create_ui()
        self.master.after(1000, self.play_ai_turn)
    
    def load_words(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return [word.strip().upper() for word in file if len(word.strip()) == 5]
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {file_path}")
            self.master.quit()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading the file: {e}")
            self.master.quit()
    
    def select_word(self):
        if not self.word_list:
            messagebox.showerror("Error", "No valid words found in the file.")
            self.master.quit()
        return random.choice(self.word_list)
    
    def create_ui(self):
        self.grid_frame = tk.Frame(self.master)
        self.grid_frame.pack(pady=20)
        
        self.grid = [[tk.Label(self.grid_frame, width=2, height=1, relief='solid', font=('Arial', 24)) 
                      for _ in range(5)] for _ in range(6)]
        
        for i in range(6):
            for j in range(5):
                self.grid[i][j].grid(row=i, column=j, padx=2, pady=2)
        
        self.ai_frame = tk.Frame(self.master)
        self.ai_frame.pack(pady=10)

        self.ai_label = tk.Label(self.ai_frame, text="AI Guess: _____", font=('Arial', 16))
        self.ai_label.pack()

        self.status_label = tk.Label(self.ai_frame, text="Game in progress...", font=('Arial', 14))
        self.status_label.pack(pady=10)
    
    def play_ai_turn(self):
        if self.current_row < 6:
            self.ai_guess = self.ai.predict(self.ai_guess)
            self.ai_label.config(text=f"AI Guess: {self.ai_guess}")
            
            for i, letter in enumerate(self.ai_guess):
                self.grid[self.current_row][i].config(text=letter)
            
            self.master.update()
            self.master.after(1000, self.check_guess)
        else:
            self.end_game(False)
    
    def check_guess(self):
        guess = self.ai_guess
        feedback = [''] * 5
        
        for i in range(5):
            if guess[i] == self.target_word[i]:
                self.grid[self.current_row][i].config(bg='green')
                feedback[i] = 'green'
            elif guess[i] in self.target_word:
                self.grid[self.current_row][i].config(bg='yellow')
                feedback[i] = 'yellow'
            else:
                self.grid[self.current_row][i].config(bg='gray')
                feedback[i] = 'gray'
        
        self.ai.update_letter_info(guess, feedback)
        self.ai.train(guess, self.target_word)
        
        if guess == self.target_word:
            self.end_game(True)
        else:
            self.current_row += 1
            self.master.after(1000, self.play_ai_turn)
    
    def end_game(self, success):
        if success:
            self.status_label.config(text=f"AI guessed the word in {self.current_row + 1} tries!")
        else:
            self.status_label.config(text=f"AI failed. The word was {self.target_word}")
        
        play_again = messagebox.askyesno("Game Over", "Do you want to play again?")
        if play_again:
            self.reset_game()
        else:
            self.master.quit()
    
    def reset_game(self):
        self.target_word = self.select_word()
        self.current_row = 0
        self.ai_guess = '_____'
        self.ai = WordleAI(self.word_list)
        
        for row in self.grid:
            for cell in row:
                cell.config(text='', bg='white')
        
        self.ai_label.config(text="AI Guess: _____")
        self.status_label.config(text="Game in progress...")
        
        self.master.after(1000, self.play_ai_turn)

if __name__ == "__main__":
    root = tk.Tk()
    game = WordleGame(root)
    root.mainloop()
