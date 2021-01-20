import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

np.random.seed(1)
np.set_printoptions(suppress=True)
model_path = "MyNN\\Saved_Model\\"
mse_train = []
mse_validation = []
rmse_train = []
rmse_validation = []


class NeuronModel:
    def __init__(self, num_of_connections=0):
        self.input_value = 0.0
        self.activation_value = 0.0
        self.output_value = 0.0
        self.error = 0.0
        self.connections = np.random.uniform(-1, 1, num_of_connections)
        self.delta_weights = np.zeros(num_of_connections)
        '''' Initialising Bias Neuron'''
        if num_of_connections == 0:
            self.input_value = 1.0
            self.activation_value = 1.0

    def __str__(self):
        string = {'input_value': self.input_value,
                  'activation_value': self.activation_value,
                  'connections': self.connections,
                  'delta_weights': self.delta_weights,
                  'error': self.error,
                  }
        # string = str(self.__dict__)
        return str(string)


class NeuronNetwork:

    def __init__(self):
        self.hidden_layer = []
        self.output_layer = []
        self.lmbd = 0.6
        self.learning_rate = 0.4
        self.alpha = 0.2
        self.min_values = None
        self.max_values = None

    def init_network(self, *args):
        for i in range(args[1]):
            self.hidden_layer.append(NeuronModel(args[0] + 1))
        self.hidden_layer.append(NeuronModel(0))
        for i in range(args[2]):
            self.output_layer.append(NeuronModel(args[1] + 1))

    def display_network(self):
        print('-------Hidden Layer------------')
        for i in self.hidden_layer:
            print(i)
        print('===============================')
        print('-------Output Layer------------')
        for i in self.output_layer:
            print(i)
        print('============complete===================')
        return ''

    '''These are called only during Training'''
    def mse_calc(self):
        temp_sum = 0.0
        for neuron in self.output_layer:
            temp_sum += neuron.error**2
        return temp_sum/2

    def delta_weight_calc(self, gradient, input_activated_value, previous_delta_weight):
        return self.learning_rate * gradient * input_activated_value + self.alpha * previous_delta_weight

    def local_output_gradient(self, output_neuron):
        return self.sigmoid(output_neuron.activation_value, derivative=True) * output_neuron.error

    def local_hidden_gradient(self, input_row):
        input_row = np.append(input_row, 1)
        for hidden_index, hidden_neuron in enumerate(self.hidden_layer[:-1]):
            sum_of_errors = 0.0
            for output_index, output_neuron in enumerate(self.output_layer):
                sum_of_errors += output_neuron.connections[hidden_index] * output_neuron.delta
            hidden_neuron.delta = sum_of_errors * self.sigmoid(hidden_neuron.activation_value, derivative=True)
            for input_cell, idx, connection in zip(input_row, range(len(hidden_neuron.connections)),
                                                   hidden_neuron.connections):
                hidden_neuron.delta_weights[idx] = self.delta_weight_calc(hidden_neuron.delta, input_cell,
                                                                          hidden_neuron.delta_weights[idx])

    def update_weights(self):
        for neuron in self.output_layer:
            neuron.connections += neuron.delta_weights
        for neuron in self.hidden_layer:
            neuron.connections += neuron.delta_weights

    '''These are called during prediction'''
    def sigmoid(self, v, derivative=False):
        if derivative:
            return self.lmbd * (v * (1 - v))
        return 1.0 / (1.0 + math.exp(-self.lmbd * v))

    def dot_prod(self, weight, input_x):
        total = 0.0
        for x, w in zip(weight, input_x):
            total += x * w
        return total

    def forward_propagation(self, input_row, output_row, predict=False):
        hidden_activated = np.zeros(len(self.hidden_layer))
        row = np.append(input_row, 1)
        row.reshape(-1, 1)
        for i, hidden_neuron in enumerate(self.hidden_layer[:-1]):
            hidden_neuron.input_value = self.dot_prod(hidden_neuron.connections, row)
            hidden_neuron.activation_value = self.sigmoid(hidden_neuron.input_value)
            hidden_activated[i] = hidden_neuron.activation_value
        hidden_activated[-1] = 1.0
        hidden_activated.reshape(-1, 1)

        prediction = []
        for output_neuron in self.output_layer:
            output_neuron.input_value = self.dot_prod(output_neuron.connections, hidden_activated)
            output_neuron.activation_value = self.sigmoid(output_neuron.input_value)
            prediction.append(output_neuron.activation_value)
        if not predict:
            self.back_propagation(input_row, output_row)
        else:
            return prediction

    def back_propagation(self, input_row, actual_output_row):
        for actual_output_value, neuron in zip(actual_output_row, self.output_layer):
            neuron.output_value = actual_output_value
            neuron.error = neuron.output_value - neuron.activation_value
            neuron.delta = self.local_output_gradient(neuron)
            for i, conn in enumerate(neuron.connections):
                neuron.delta_weights[i] = self.delta_weight_calc(neuron.delta, self.hidden_layer[i].activation_value,
                                                                 neuron.delta_weights[i])
        self.local_hidden_gradient(input_row)
        self.update_weights()

    def train_nn(self, x_train, y_train, epoch, x_test, y_test, patience, min_delta):
        temp_patience = 0
        epoch_for_graph = 0
        for t in range(1, epoch + 1):
            mse_train_temp = 0.0
            for row_x, row_y in zip(x_train, y_train):
                self.forward_propagation(row_x, row_y)
                mse_train_temp += self.mse_calc()
            print(' . ', t)
            epoch_for_graph += 1
            mse_train.append(mse_train_temp / (x_train.shape[0]))
            rmse_train.append(math.sqrt(mse_train_temp / (x_train.shape[0])))
            self.validate_nn(x_test, y_test)
            if t >= 2 and (mse_validation[-1] - mse_validation[-2]) >= min_delta:
                temp_patience += 1
                #print("Val Loss increasing... Patience:", temp_patience)
            if temp_patience == patience:
                #print("Stopping Criteria met. Stopping Training")
                break

        # print("Training has been completed.")
        # print("Train MSE: ", mse_train[-1])
        # print("Train RMSE: ", rmse_train[-1])
        # print("Validation MSE: ", mse_validation[-1])
        # print("Validation RMSE: ", rmse_validation[-1])

        self.plot_graph(epoch_for_graph)

    def validate_nn(self, x_test, y_test):
        mse_val_temp = 0.0
        for row_x, row_y in zip(x_test, y_test):
            self.forward_propagation(row_x, row_y, predict=True)
            for i in range(len(self.output_layer)):
                self.output_layer[i].output_value = row_y[i]
                self.output_layer[i].error = self.output_layer[i].output_value - self.output_layer[i].activation_value
            mse_val_temp += self.mse_calc()
        mse_validation.append(mse_val_temp / x_test.shape[0])
        rmse_validation.append(math.sqrt(mse_val_temp/x_test.shape[0]))

    def plot_graph(self, epoch):
        epochs = range(0, epoch)
        plt.plot(epochs, rmse_train, 'g', label='Training RMSE')
        plt.plot(epochs, rmse_validation, 'b', label='Validation RMSE')
        plt.title('Training and Validation RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.show()

    def save_model(self):
        output_weights = []
        hidden_weights = []
        for neuron in self.output_layer:
            output_weights.append(neuron.connections)
        for neuron in self.hidden_layer:
            hidden_weights.append(neuron.connections)

        hidden_file_obj = open(model_path+'_hidden.pkl', 'wb')
        output_file_obj = open(model_path + '_output.pkl', 'wb')
        pickle.dump(hidden_weights, hidden_file_obj)
        pickle.dump(output_weights, output_file_obj)
        hidden_file_obj.close()
        output_file_obj.close()

        print("Model Saved.")

    def load_model(self, num_of_input_neurons):
        #hidden_file_obj = open(model_path + '_hidden.pkl', 'rb')
        #output_file_obj = open(model_path + '_output.pkl', 'rb')
        # hidden_file_obj = open(model_path + '_hidden_old_best.pkl', 'rb')
        # output_file_obj = open(model_path + '_output_old_best.pkl', 'rb')
        hidden_file_obj = open(model_path + 'mine11_12_hidden.pkl', 'rb')
        output_file_obj = open(model_path + 'mine11_12_output.pkl', 'rb')

        saved_hidden_weights = pickle.load(hidden_file_obj)
        saved_output_weights = pickle.load(output_file_obj)
        hidden_file_obj.close()
        output_file_obj.close()

        self.init_network(num_of_input_neurons, len(saved_hidden_weights) - 1, len(saved_output_weights))
        for neuron, item in zip(self.hidden_layer[:-1], saved_hidden_weights):
            neuron.connections = item
        for neuron, item in zip(self.output_layer, saved_output_weights):
            neuron.connections = item
        print("\n -----------Saved Model has been Loaded----------")
        self.display_network()

    def predict(self, input_value):
        print("-------------Now predicting......")
        result = self.forward_propagation(input_value, None, predict=True)
        return np.array(result)

    def normalised(self, x):
        return (x - self.max_values[0:2])/(self.max_values[0:2]-self.min_values[0:2])

    def de_normalised(self, y):
        return ((self.max_values[2:] - self.min_values[2:]) * y) + self.min_values[2:]
