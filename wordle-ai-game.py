import tkinter as tk
from tkinter import messagebox
import random
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
import requests
import json

class OllamaLlama:
    def __init__(self, base_url="http://localhost:11434", word_list=None):
        self.base_url = base_url
        self.model = "llama3.1:8b"
        self.word_list = word_list or []
        self.excluded_letters = set()
        self.included_letters = set()
        self.letter_positions = {i: set() for i in range(5)}
        self.valid_words = set(self.word_list)

    def generate(self, prompt, max_tokens=100):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

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

    def get_suggestion(self):
        prompt = f"""
        I'm playing Wordle. Here's what I know:
        Excluded letters: {', '.join(sorted(self.excluded_letters))}
        Included letters: {', '.join(sorted(self.included_letters))}
        Letter positions:
        {' '.join([''.join(sorted(self.letter_positions[i])) or '_' for i in range(5)])}
        
        Based on this information, suggest a 5-letter word that could be the answer.
        The word must be in the following list of valid words: {', '.join(self.valid_words)}
        Only provide the word, no additional explanation.
        """
        suggestion = self.generate(prompt)
        return suggestion.strip().upper() if suggestion else None

class WordleAI:
    def __init__(self, word_list, embedding_dim=50, hidden_dim=100):
        self.word_list = word_list
        self.vocab_size = len(word_list)
        self.embedding_dim = embedding_dim
        self.embedding = Linear(26, embedding_dim)
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
        self.master.title("Wordle with AI Learning")
        self.master.geometry("300x700")  # Increased height to accommodate new elements
        
        self.word_list = self.load_words(r"C:\Users\danik\Desktop\wordle\words.txt")
        self.target_word = self.select_word()
        self.current_row = 0
        self.current_col = 0
        
        self.ai = WordleAI(self.word_list)
        self.ollama = OllamaLlama(word_list=self.word_list)
        self.ai_guess = '_____'
        self.ollama_guess = '_____'
        
        self.create_ui()
    
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
        
        self.keyboard_frame = tk.Frame(self.master)
        self.keyboard_frame.pack(pady=10)
        
        keyboard_layout = [
            "QWERTYUIOP",
            "ASDFGHJKL",
            "ZXCVBNM"
        ]
        
        self.key_buttons = {}
        
        for row, keys in enumerate(keyboard_layout):
            key_frame = tk.Frame(self.keyboard_frame)
            key_frame.pack()
            
            if row == 2:
                self.enter_button = tk.Button(key_frame, text="Enter", width=4, command=self.check_guess)
                self.enter_button.pack(side=tk.LEFT, padx=1, pady=1)
            
            for key in keys:
                self.key_buttons[key] = tk.Button(key_frame, text=key, width=2, 
                                                  command=lambda x=key: self.key_press(x))
                self.key_buttons[key].pack(side=tk.LEFT, padx=1, pady=1)
            
            if row == 2:
                self.backspace_button = tk.Button(key_frame, text="⌫", width=4, command=self.backspace)
                self.backspace_button.pack(side=tk.LEFT, padx=1, pady=1)

        self.ai_frame = tk.Frame(self.master)
        self.ai_frame.pack(pady=10)

        self.ai_label = tk.Label(self.ai_frame, text="TinyGrad AI Guess: _____", font=('Arial', 12))
        self.ai_label.pack()

        self.ollama_label = tk.Label(self.ai_frame, text="Ollama AI Guess: _____", font=('Arial', 12))
        self.ollama_label.pack()

        self.ai_button = tk.Button(self.ai_frame, text="Get TinyGrad AI Suggestion", command=self.get_ai_suggestion)
        self.ai_button.pack()

        self.ollama_button = tk.Button(self.ai_frame, text="Get Ollama AI Suggestion", command=self.get_ollama_suggestion)
        self.ollama_button.pack()
    
    def key_press(self, key):
        if self.current_col < 5:
            self.grid[self.current_row][self.current_col].config(text=key)
            self.current_col += 1
    
    def backspace(self):
        if self.current_col > 0:
            self.current_col -= 1
            self.grid[self.current_row][self.current_col].config(text='')
    
    def check_guess(self):
        if self.current_col == 5:
            guess = ''.join(self.grid[self.current_row][i]['text'] for i in range(5))
            
            if guess not in self.word_list:
                messagebox.showwarning("Invalid Word", "Please enter a valid word.")
                return
            
            feedback = [''] * 5
            for i in range(5):
                if guess[i] == self.target_word[i]:
                    self.grid[self.current_row][i].config(bg='green')
                    self.key_buttons[guess[i]].config(bg='green')
                    feedback[i] = 'green'
                elif guess[i] in self.target_word:
                    self.grid[self.current_row][i].config(bg='yellow')
                    if self.key_buttons[guess[i]]['bg'] != 'green':
                        self.key_buttons[guess[i]].config(bg='yellow')
                    feedback[i] = 'yellow'
                else:
                    self.grid[self.current_row][i].config(bg='gray')
                    self.key_buttons[guess[i]].config(bg='gray')
                    feedback[i] = 'gray'
            
            self.ai.update_letter_info(guess, feedback)
            self.ollama.update_letter_info(guess, feedback)
            self.ai.train(guess, self.target_word)

            self.ai_guess = self.ai.predict(guess)
            self.ollama_guess = self.ollama.get_suggestion()
            self.ai_label.config(text=f"TinyGrad AI Guess: {self.ai_guess}")
            self.ollama_label.config(text=f"Ollama AI Guess: {self.ollama_guess}")
            
            if guess == self.target_word:
                messagebox.showinfo("Congratulations!", "You guessed the word!")
                self.master.quit()
            elif self.current_row == 5:
                messagebox.showinfo("Game Over", f"The word was {self.target_word}")
                self.master.quit()
            else:
                self.current_row += 1
                self.current_col = 0

    def get_ai_suggestion(self):
        suggestion = self.ai.predict(self.ai_guess)
        self.ai_guess = suggestion
        self.ai_label.config(text=f"TinyGrad AI Guess: {self.ai_guess}")
        messagebox.showinfo("TinyGrad AI Suggestion", f"TinyGrad AI suggests: {suggestion}")

    def get_ollama_suggestion(self):
        suggestion = self.ollama.get_suggestion()
        if suggestion:
            self.ollama_guess = suggestion
            self.ollama_label.config(text=f"Ollama AI Guess: {suggestion}")
            messagebox.showinfo("Ollama AI Suggestion", f"Ollama AI suggests: {suggestion}")
        else:
            messagebox.showwarning("Ollama AI Error", "Failed to get a suggestion from Ollama AI")

if __name__ == "__main__":
    root = tk.Tk()
    game = WordleGame(root)
    root.mainloop()
