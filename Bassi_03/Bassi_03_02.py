# Bassi, Aman
# 1001-393-217
# 2017-10-09
# Assignment_03_02
from __future__ import division

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import random
import Bassi_03_03 as b3
import tkinter as tk

class DisplayActivationFunctions:
    """
    This class is for displaying activation functions for NN.
    Aman Bassi 2017_10_09
    """

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.batch_size = 100
        self.xmin = 0
        self.xmax = 1000
        self.delayed_elements = 10
        self.iterations = 10
        self.ymin = 0
        self.ymax = 2
        self.training_size = 80
        self.learning_rate = 0.1
        self.activation_function = "Sigmoid"
        self.mae_price = []
        self.mae_volume = []
        self.mse_price = []
        self.mse_volume = []
        self.predict_price = []
        self.predict_volume = []
        self.mae_max_price = []
        self.mae_max_volume = []
        self.mse_mean_price = []
        self.mse_mean_volume = []
        self.f, self.axis = plt.subplots(2,2)
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        # self.axes.set_xlabel('Input')
        # self.axes.set_ylabel('Output')
        # self.axes.set_title("")
        # plt.xlim(self.xmin, self.xmax)
        # plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0,  sticky=tk.N + tk.W + tk.S + tk.E)
        self.sliders_frame.rowconfigure(0, weight=5)
        self.sliders_frame.rowconfigure(1, weight=5)
        self.sliders_frame.rowconfigure(2, weight=5)

        self.sliders_frame.columnconfigure(0, weight=5)
        # set up the sliders
        self.delayed_elements_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                                from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                                activebackground="#FF0000",
                                                highlightcolor="#00FFFF",
                                                label="delayed elements",
                                                command=lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.set(self.delayed_elements)
        self.delayed_elements_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.learning_rate_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Alpha",
                                            command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.training_size_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                             from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="training size",
                                             command=lambda event: self.training_size_slider_callback())
        self.training_size_slider.set(self.training_size)
        self.training_size_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.sliders_frame1 = tk.Frame(self.master)
        self.sliders_frame1.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=5)
        self.sliders_frame.rowconfigure(1, weight=5)
        self.sliders_frame.columnconfigure(0, weight=5)

        self.batch_size_slider = tk.Scale(self.sliders_frame1, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                    from_=1, to_=200, resolution=1, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Batch Size",
                                    command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.grid(row=0, column=0, columnspan =2,sticky=tk.N + tk.E + tk.S + tk.W)

        self.iterations_slider = tk.Scale(self.sliders_frame1, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                          from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF",
                                          label="iterations",
                                          command=lambda event: self.iterations_slider_callback())
        self.iterations_slider.set(self.iterations)
        self.iterations_slider.grid(row=1, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1)
        self.buttons_frame.rowconfigure(1, weight=1)
        self.buttons_frame.rowconfigure(2, weight=1)
        self.buttons_frame.rowconfigure(3, weight=15)
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.weightsbutton = tk.Button(self.buttons_frame, text="set_to_zero", command=self.adjust_to_zero)
        self.weightsbutton.grid(row=1, column=0, sticky=tk.N + tk.E + tk.W + tk.S)

        self.adjust_weights = tk.Button(self.buttons_frame, text="adjust_weights", command=self.adjust_weights)
        self.adjust_weights.grid(row=2, column=0, sticky=tk.N + tk.E + tk.W + tk.S)

        #normalized data
        self.data = b3.reading_data("data.txt")
        print(self.data[8, :1])
        # print("data",self.data.shape)

        #getting test data and train data after partitioning the data
        self.train_data, self.test_data = self.partitioning(self.data)
        # print(self.train_data.shape, self.test_data)


        # self.input = np.append(self.data[0:8, 0], self.data[0:8, 1])
        # self.input = np.append(self.input, 1)
        # self.input = self.input.reshape((17, 1))


        # self.display_activation_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

    def display_activation_function(self):

        self.axes.cla()
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        self.axis[0, 0].plot(range(len(self.mae_max_price)), self.mae_max_price)
        self.axis[0, 0].set_title('Mae price')
        self.axis[0, 0].set_ylim([0, 2])

        self.axis[0, 1].plot(range(len(self.mae_max_volume)), self.mae_max_volume)
        self.axis[0, 1].set_title('Mae Volume')
        self.axis[0, 1].set_ylim([0, 2])

        self.axis[1, 0].plot(range(len(self.mse_mean_price)), self.mse_mean_price)
        self.axis[1, 0].set_title('Mse Price')
        self.axis[1, 0].set_ylim([0, 2])

        self.axis[1, 1].plot(range(len(self.mse_mean_volume)), self.mse_mean_volume)
        self.axis[1, 1].set_title('Mse Volume')
        self.axis[1, 1].set_ylim([0, 2])

        # plt.xlim(self.xmin, self.xmax)
        # plt.ylim(self.ymin, self.ymax)
        # plt.title(self.activation_function)
        # plt.plot(range(len(self.mae_max_price)), self.mae_max_price)
        # plt.plot(range(len(self.mae_max_volume)), self.mae_max_volume)
        # plt.plot(range(len(self.mse_mean_price)), self.mse_mean_price)
        # plt.plot(range(len(self.mse_mean_volume)), self.mse_mean_volume)
        self.canvas.draw()

    def training_size_slider_callback(self):
        self.training_size = self.training_size_slider.get()

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()
        self.display_activation_function()

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()

    def iterations_slider_callback(self):
        self.iterations = self.iterations_slider.get()

    def delayed_elements_slider_callback(self):
        self.delayed_elements = self.delayed_elements_slider.get()

    def adjust_to_zero(self):
        self.mae_price = []
        self.mae_volume = []
        self.mse_price = []
        self.mse_volume = []
        self.predict_price = []
        self.predict_volume = []
        self.mae_max_price = []
        self.mae_max_volume = []
        self.mse_mean_price = []
        self.mse_mean_volume = []
        self.axis[0, 0].cla()
        self.axis[0, 1].cla()
        self.axis[1, 0].cla()
        self.axis[1, 1].cla()
        m = int(2*self.delayed_elements + 3)
        self.weights = np.zeros((2, m))
        # print(self.weights, self.weights.shape)

    def adjust_weights(self):
        input_name = np.zeros((int(2*self.delayed_elements + 3), 1))
        target = np.zeros((1, 2))
        number_of_batches = int(len(self.train_data)/self.batch_size)
        for iter in range(int(self.iterations)):
            # print("iter",iter)
            batch_size = self.batch_size
            j = int(self.delayed_elements + 1)
            counter = 1
            for batch in range(number_of_batches):
                l = batch*self.batch_size
                # print("batch", batch)
                counter = counter + 1
                # print(" counter ", counter)
                for i in range(int(l), int(batch_size - self.delayed_elements)):
                    if i < batch_size-self.delayed_elements-1:
                        # print("my num", i)
                        a = self.train_data[i:j, 0]
                        input_name[0: int(self.delayed_elements + 1), 0] = a
                        input_name[int(self.delayed_elements + 1):(len(input_name)-1), 0] = self.train_data[i:j, 1]
                        input_name[len(input_name)-1, 0] = 1
                        # print("hello")
                        # print(input_name)
                        net_val = np.dot(self.weights, input_name)
                        activation = net_val
                        # print("act", activation.shape)
                        target[0, :2] = self.train_data[j, :2]
                        error = target.transpose() - activation
                        # print("error", error.shape)
                        self.weights_update(input_name, error)
                        # if()
                        # print(j)
                        j = j + 1
                    else:
                        # print(i, j)
                        self.classify()
                        self.display_activation_function()
                        j = int(batch_size + self.delayed_elements + 1)
                        # print(j)
                        batch_size = counter*self.batch_size
                        break
        print(self.mae_max_volume,self.mae_max_price, self.mse_mean_volume,self.mse_mean_price)


    def weights_update(self, input_name, error):
        a = 2*self.learning_rate*error
        change = np.dot(a, input_name.transpose())
        self.weights = self.weights + change


    def classify(self):
        self.mae_price = []
        self.mae_volume = []
        self.mse_price = []
        self.mse_volume = []

        test_data = np.zeros((int(2 * self.delayed_elements + 3), 1))
        # print(test_data.shape)
        test_data[len(test_data)-1, 0] = 1
        # print(test_data)
        j = int(self.delayed_elements+1)
        for i in range(len(self.test_data)):
            if i < len(self.test_data)-self.delayed_elements-1:
                # print(self.test_data[0:8, 0])
                a = self.test_data[i:j, 0]
                test_data[0:int(self.delayed_elements+1), 0] = a
                test_data[int(self.delayed_elements+1):int(len(test_data)-1), 0] = self.test_data[i:j, 1]
                target_price = self.test_data[j, 0]
                target_volume = self.test_data[j, 1]
                target_values = np.dot(self.weights, test_data)
                predicted_price = target_values[0, 0]
                predicted_volume = target_values[1, 0]
                # print("classify", i)
                self.predict_price.append(predicted_price)
                self.predict_volume.append(predicted_volume)
                j = j + 1
                self.error_price(target_price, predicted_price)
                self.error_volume(target_volume, predicted_volume)
                self.mean_error_price(target_price, predicted_price)
                self.mean_error_volume(target_volume, predicted_volume)
        self.mae_max_price.append(max(self.mae_price))
        self.mae_max_volume.append(max(self.mae_volume))
        self.mse_mean_price.append(np.mean(self.mse_price))
        self.mse_mean_volume.append(np.mean(self.mse_volume))


    def error_price(self, target_price, predicted_price):
        mae_price = abs(target_price - predicted_price)
        self.mae_price.append(mae_price)

    def error_volume(self, target_volume, predicted_volume):
        mae_volume = abs(target_volume - predicted_volume)
        self.mae_volume.append(mae_volume)

    def mean_error_price(self, target_price, predicted_price):
        mse_price = pow((target_price - predicted_price), 2)
        self.mse_price.append(mse_price)

    def mean_error_volume(self, target_volume, predicted_volume):
        mse_volume = pow((target_volume - predicted_volume), 2)
        self.mse_volume.append(mse_volume)

    def partitioning(self, data):
        m_train = int((self.training_size * len(data))/100)
        train_data = data[:m_train, :]
        if self.training_size != 100:
            test_data = data[m_train:len(data), :]
        else:
            test_data = []

        return train_data, test_data