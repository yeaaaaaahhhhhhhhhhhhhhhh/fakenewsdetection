import tkinter as tk
from tkinter import messagebox
import math
import re
import json
from collections import defaultdict


def load_corpus(filename):
    data = []
    try:
        with open(filename, "r", encoding="utf-8") as jsonfile:
            corpus = json.load(jsonfile)
            for label, articles in corpus.items():
                for text in articles:
                    data.append((text, label))
    except FileNotFoundError:
        messagebox.showerror("Error", f"File '{filename}' not found!")
    return data


def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words


class NaiveBayesClassifier:
    def __init__(self):
        self.classes = {}
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_totals = defaultdict(int)
        self.vocab = set()
        self.total_documents = 0

    def train(self, data):
        self.total_documents = len(data)
        for text, label in data:
            self.classes[label] = self.classes.get(label, 0) + 1
            words = tokenize(text)
            for word in words:
                self.word_counts[label][word] += 1
                self.class_totals[label] += 1
                self.vocab.add(word)

    def predict(self, text):
        words = tokenize(text)
        log_probs = {}

        for label in self.classes:
            # Prior probability
            log_prob = math.log(self.classes[label] / self.total_documents)
            for word in words:
                word_count = self.word_counts[label].get(word, 0)
                # Laplace smoothing
                log_prob += math.log((word_count + 1) / (self.class_totals[label] + len(self.vocab)))
            log_probs[label] = log_prob

        # Decide classification
        best_label = max(log_probs, key=log_probs.get)
        return best_label, log_probs


training_data = load_corpus("corpus.json")
if not training_data:
    print("No training data loaded! Please check your corpus.json file.")
    exit()

# Initialize and train model
model = NaiveBayesClassifier()
model.train(training_data)


# TKINTER UI
def check_news():
    article = text_input.get("1.0", tk.END).strip()
    if not article:
        messagebox.showwarning("Warning", "Please enter a news article!")
        return

    result, scores = model.predict(article)

    # Format output scores
    legit_score = scores.get("Legit", 0)
    fake_score = scores.get("Fake", 0)

    result_message = (
        f"Article Classified as: {result}\n\n"
        f"Legit Score: {legit_score:.4f}\n"
        f"Fake Score: {fake_score:.4f}"
    )

    messagebox.showinfo("Result", result_message)

# Create UI window
root = tk.Tk()
root.title("Fake News Detection - Naive Bayes")
root.geometry("500x400")

# UI Elements
label = tk.Label(root, text="Enter News Article:", font=("Arial", 12))
label.pack(pady=10)

text_input = tk.Text(root, height=8, width=50)
text_input.pack(pady=10)

check_button = tk.Button(root, text="Check if Fake or Legit", command=check_news, font=("Arial", 12), bg="lightblue")
check_button.pack(pady=10)

root.mainloop()
