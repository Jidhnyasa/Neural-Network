# Bassi, Aman
# 1001-393-217
# 2017-09-17
# Assignment_01_01
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import Bassi_01_02 as k02  # This module is for plotting components


class WidgetsWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # Create a frame for plotting graphs
        self.left_frame = tk.Frame(self, bg='red')
        self.display_activation_functions = k02.DisplayActivationFunctions(self, self.left_frame)
        # Create a frame for displaying graphics
        self.left_frame.grid(row=0, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.left_frame.rowconfigure(0,weight=1)
        self.left_frame.columnconfigure(0, weight=1)
        # self.left_frame.grid_propagate(True)

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

widgets_window = WidgetsWindow()
# widgets_window.geometry("500x500")
# widgets_window.wm_state('zoomed')
widgets_window.title('Assignment_01 --  Bassi')
widgets_window.minsize(600,300)
widgets_window.protocol("WM_DELETE_WINDOW", lambda root_window=widgets_window: close_window_callback(root_window))
widgets_window.mainloop()