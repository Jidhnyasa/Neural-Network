# Bassi, Aman
# 1001-393-217
# 2017-09-27
# Assignment_02_01

import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import Bassi_02_02 as k02  # This module is for plotting components



class WidgetsWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.center_frame = tk.Frame(self)
        # Create a frame for plotting graphs
        self.display_activation_functions = k02.DisplayActivationFunctions(self, self.center_frame)
        # Create a frame for displaying graphics
        self.center_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)




def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

widgets_window = WidgetsWindow()
# widgets_window.geometry("500x500")
# widgets_window.wm_state('zoomed')
widgets_window.title('Assignment_02 --  Bassi')
widgets_window.minsize(600,300)
widgets_window.protocol("WM_DELETE_WINDOW", lambda root_window=widgets_window: close_window_callback(root_window))
widgets_window.mainloop()