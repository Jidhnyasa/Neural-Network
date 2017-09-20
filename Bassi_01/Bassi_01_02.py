# Bassi, Aman
# 1001-393-217
# 2017-09-17
# Assignment_01_02

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import tkinter as tk
class DisplayActivationFunctions:
    """
    This class is for displaying activation functions for NN.
    Aman Bassi 2017_09_17
    """
    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.x = []
        self.a = [0,0,0,0]
        self.y = []
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight = 1
        self.input_weight2 = 1
        self.bias = 0
        self.activation_function = "Symmetric Hard Limit"
        self.check_random=0
        self.check_train=0
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=1)
        self.sliders_frame.rowconfigure(1, weight=1)
        self.sliders_frame.rowconfigure(2, weight=1)
        self.sliders_frame.columnconfigure(0, weight=1)
        # self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        self.input_weight_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight",
                                            command=lambda event: self.input_weight_slider_callback())
        self.input_weight_slider.set(self.input_weight)
        self.input_weight_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.input_weight2_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight2",
                                            command=lambda event: self.input_weight2_slider_callback())
        self.input_weight2_slider.set(self.input_weight2)
        self.input_weight2_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1,columnspan=2,sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1)
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Transfer Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetric Hard Limit", "Linear","Hyperbolic Tangent",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetric Hard Limit")
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_button=tk.Button(self.buttons_frame,text="train", command=self.train_data)
        self.activation_function_button.grid(row=2,column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_button2 = tk.Button(self.buttons_frame, text="random data",command=self.random_data)
        self.activation_function_button2.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())





            # self.activation_function_button.bind("<Button-1>", train_data)

    def display_activation_function(self):
        boundary=10
        classify=1
        input_values = np.linspace(-10, 10,256, endpoint=True)
        input_values2=np.linspace(-10,10,256,endpoint=True)
        self.axes.cla()
        self.axes.cla()
        self.axes.plot()
        self.axes.xaxis.set_visible(True)
        if self.check_random == 1:
            plt.plot([self.x[0], self.x[1]], [self.y[0], self.y[1]], 'bo')
            plt.plot([self.x[2], self.x[3]], [self.y[2], self.y[3]], 'k^')
        if self.check_train == 1:
            line_y=[0,0,0,0]
            line_y[0]=(-self.bias-self.input_weight * self.x[0])/self.input_weight2
            line_y[1] = (-self.bias - self.input_weight * self.x[1]) / self.input_weight2
            line_y[2] = (-self.bias - self.input_weight * self.x[2]) / self.input_weight2
            line_y[3] = (-self.bias - self.input_weight * self.x[3]) / self.input_weight2
            if((self.y[0]>line_y[0] and self.y[1]>line_y[1])):
                boundary,classify = 10,-1
            elif((self.y[0]<line_y[0] and self.y[1])<line_y[1]):
                boundary, classify =-10,-1

            if ((self.y[2] > line_y[2] and self.y[3] > line_y[3])):
                boundary, classify = 10, 1
            elif ((self.y[2] < line_y[2] and self.y[3]) < line_y[3]):
                boundary, classify = -10, 1


            if(classify==1):
                plt.fill_between(input_values, (-self.bias - self.input_weight*input_values)/self.input_weight2,boundary,color="green")
                plt.fill_between(input_values, (-self.bias - self.input_weight * input_values) / self.input_weight2,
                                 -boundary, color="red")
            if (classify == -1):
                plt.fill_between(input_values, (-self.bias - self.input_weight*input_values)/self.input_weight2, boundary, color="red")
                plt.fill_between(input_values, (-self.bias - self.input_weight * input_values) / self.input_weight2,
                                 -boundary, color="green")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title(self.activation_function)

        self.canvas.draw()


    def input_weight2_slider_callback(self):
        self.input_weight2 = self.input_weight2_slider.get()
        self.train_data()

    def input_weight_slider_callback(self):
        self.input_weight = self.input_weight_slider.get()
        self.train_data()

    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        self.train_data()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.train_data()

    def random_data(self):
        self.x=[]
        self.y=[]
        self.check_random=1
        for i in range(4):
            self.x.append(random.randint(-10, 10))
            self.y.append(random.randint(-10, 10))
            print(self.x[i], self.y[i])
        self.check_train = 0
        self.display_activation_function()

    def train_data(self):
        if self.check_random == 1 and self.check_train == 0:
            self.input_weight = self.input_weight_slider.get()
            self.input_weight2 = self.input_weight2_slider.get()
            self.bias = self.bias_slider.get()
            t = [-1, -1, 1, 1]
            for i in range(100):
                for j in range(4):
                    net_value = self.input_weight * self.x[j] + self.input_weight2 * self.y[j] + self.bias
                    if self.activation_function == 'Symmetric Hard Limit':
                        if net_value >= 0:
                            activation = 1
                        else:
                            activation = -1
                    elif self.activation_function == "Linear":
                        activation = net_value
                    elif self.activation_function == "Hyperbolic Tangent":
                        activation = (np.exp(net_value) - np.exp(-net_value)) / (np.exp(net_value) + np.exp(-net_value))

                    w_new = self.input_weight+(t[j]-activation)*self.x[j]
                    w_new2 = self.input_weight2 + (t[j]-activation)*self.y[j]
                    b_new = self.bias + (t[j]-activation)
                    print (w_new,w_new2)
                    self.input_weight = w_new
                    self.input_weight2 = w_new2
                    self.bias = b_new
                    self.check_train = 1
                    if activation >= 0:
                        self.a[j] = 1
                    else:
                        self.a[j] = -1
                if i%30 == 0:
                    self.display_activation_function()

        elif self.check_random == 1 and self.check_train == 1:
            self.input_weight = self.input_weight_slider.get()
            self.input_weight2 = self.input_weight2_slider.get()
            self.bias = self.bias_slider.get()
            self.check_train = 0
            self.train_data()
        self.display_activation_function()

