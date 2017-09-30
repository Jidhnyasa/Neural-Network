# Bassi, Aman
# 1001-393-217
# 2017-09-27
# Assignment_02_02

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
import Bassi_02_03 as b03
import tkinter as tk
class DisplayActivationFunctions:
    """
    This class is for displaying activation functions for NN.
    Aman Bassi 2017_09_27
    """

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.random = 0
        self.xmin = 1
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 100
        self.alpha = 0.1
        self.gamma = 0.1
        self.epoch = [i for i in range(1000)]
        self.errorRate = []
        self.learning_rule = "Delta Rule"

        self.activation_function = "Hyperbolic Tangent"
        self.weights = np.zeros((10, 785))
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
        self.sliders_frame.columnconfigure(0, weight=1)
        self.sliders_frame.columnconfigure(1, weight=1)
        # set up the sliders and button
        self.alpha_learning_rate = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-0.001, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Alpha(Learning rate)",
                                            command=lambda event: self.alpha_learning_rate_callback())
        self.alpha_learning_rate.set(self.alpha)
        self.alpha_learning_rate.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.adjust_weights = tk.Button(self.sliders_frame, text="Adjust Weights(Learn)", command=self.training_data)
        self.adjust_weights.grid(row=1, column=0, sticky=tk.W)
        self.randomize_weights = tk.Button(self.sliders_frame, text="Randomize Weights", command=self.randomized_weights)
        self.randomize_weights.grid(row=1, column=1, sticky=tk.W)
        #########################################################################
        #  Set up the frame for drop_down menu and labels(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.rowconfigure(1, weight=1)
        self.buttons_frame.rowconfigure(2, weight=1)
        self.buttons_frame.rowconfigure(3, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1)
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Transfer Functions",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetric Hard Limit", "Linear", "Hyperbolic Tangent",command=lambda
                event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set(self.activation_function)
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.label_for_learning_rule = tk.Label(self.buttons_frame, text="Select Learning Method",
                                                justify="center")
        self.label_for_learning_rule.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.learning_rule_variable = tk.StringVar()
        self.learning_rule_dropdown = tk.OptionMenu(self.buttons_frame, self.learning_rule_variable, "Filtered Learning", "Delta Rule", "Unsupervised Hebb",
                                                    command=lambda
                                                        event: self.learning_rule_dropdown_callback())
        self.learning_rule_variable.set(self.learning_rule)
        self.learning_rule_dropdown.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)


        # getting input images and images actual values from b03 module dimension of image is 784*1000
        self.images_input, self.images_digit = b03.read_one_image_and_convert_to_vector("Data")
        # print(self.images_input.shape)
        # appending bias to our image_input and now dimension is 785*1000
        self.images_input = np.append(self.images_input, np.ones((1, 1000)), axis=0)
        # print(self.images_input, self.images_input.shape)
        self.images_input = np.divide(self.images_input, 127.5)
        self.images_input =np.subtract(self.images_input, 1)
        self.train_data, self.test_data, self.indices = b03.getting_data(self.images_input)
        # print(self.indices)
        # print(len(self.train_data[0]))
        # dimension is 10 * 1000
        self.target_vector = self.getting_target(self.indices, self.images_digit)
        # print(self.target_vector)
        # self.display_activation_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())


    def display_activation_function(self):

        self.axes.cla()
        self.axes.cla()
        self.axes.plot()
        self.axes.xaxis.set_visible(True)
        plt.plot(self.epoch[:len(self.errorRate)], self.errorRate)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title(self.activation_function)
        self.canvas.draw()

    def alpha_learning_rate_callback(self):
        self.alpha = self.alpha_learning_rate.get()
        # self.training_data()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        # self.training_data()

    def learning_rule_dropdown_callback(self):
        self.learning_rule= self.learning_rule_variable.get()
        # self.training_data()

    def training_data(self):
        if self.random == 1:
            activation_val = np.zeros((10,1))
            # activation_val1 = np.zeros((10, 1))
            for i in range(100):
                for j in range(len(self.train_data[0])):

                    a = self.train_data[:, j]
                    a = a.reshape((785, 1))
                    b = self.target_vector[:, j]
                    b = b.reshape((10, 1))
                    # print(b.shape)
                    # print(b, self.indices[799], len(self.train_data[0]))
                    net_value = np.dot(self.weights, a)


                    # print(net_value.shape)
                    # print(j)
                    if self.activation_function == "Symmetric Hard Limit":
                        for l in range(10):
                            if net_value[l, 0] >= 0:
                                activation_val[l, 0] = 1
                            elif net_value[l, 0] < 0:
                                activation_val[l, 0] = -1

                    elif self.activation_function == "Linear":
                        activation_val = net_value

                    elif self.activation_function == "Hyperbolic Tangent":
                        activation_val = np.tanh(net_value)

                    e = b - activation_val
                    if self.learning_rule == "Delta Rule":
                        self.weights = self.weights + np.dot((self.alpha*e), (a.transpose()))
                    elif self.learning_rule == "Filtered Learning":
                        self.weights = ((1 - self.gamma) * self.weights) + np.dot((self.alpha * (b)), (a.transpose()))
                    elif self.learning_rule == "Unsupervised Hebb":
                        self.weights = self.weights + np.dot((self.alpha * (activation_val)), (a.transpose()))
                self.classification()
                self.display_activation_function()

    def randomized_weights(self):
        self.random = 1
        self.errorRate = []
        self.weights = np.random.uniform(-0.001, 0.001, (10, 785))
        # print(self.weights, self.weights.shape, np.amax(self.weights), np.amin(self.weights))


    def getting_target(self,indices,images_digit):
        targetvector = np.zeros((10, 1000))
        for j in range(1000):
            targetvector[images_digit[indices[j]], j] = 1

        for i in range(10):
            for j in range(1000):
                if targetvector[i ,j] != 1:
                    targetvector[i,j] =-1

        return targetvector

    def classification(self):
        predict_val = np.dot(self.weights, self.test_data)
        predicted_indices = np.argmax(predict_val, axis =0)

        actual_indices =[]
        for j in range(800 , 1000):
            actual_indices.append(self.images_digit[self.indices[j]])

        # print(actual_indices)
        # print(predicted_indices)

        count = 0
        for i in range(200):
            if predicted_indices[i] == actual_indices[i]:
                count =count + 1

        self.errorRate.append(((200-count) / 200)*100)
        # print(self.errorRate)
