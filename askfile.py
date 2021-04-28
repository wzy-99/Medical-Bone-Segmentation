import tkinter as tk
from tkinter import filedialog

def ask_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def ask_dir():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory()


if __name__ == '__main__':
    print(ask_file())
    print(ask_dir())
