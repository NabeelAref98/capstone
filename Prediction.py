import numpy
import plotly.graph_objs
import sklearn
import sklearn.linear_model

#Class That does the Linear Regression prediction
class Prediction:
    #initializes the required variables
    def __init__(self, data_object):
        self.data_object = data_object
        self.index_axis = numpy.array(self.data_object.get_all_values().index).reshape(-1, 1)
        self.price_axis = self.data_object.get_all_prices()
    
    #preforms prediction trends
    def linearReg(self):
        index_train, index_test, price_train, price_test = sklearn.model_selection.train_test_split(self.index_axis, self.price_axis, test_size=0.4, random_state=100)

        # does the scaling
        sklearn.preprocessing.StandardScaler().fit(index_train)

        # Developes the model
        learning_module = sklearn.linear_model.LinearRegression()
        learning_module.fit(index_train, price_train)
        x = plotly.graph_objs.Scatter(x=index_train.T[0], y=price_train, mode="markers", name="Actual")
        y = plotly.graph_objs.Scatter(x=index_train.T[0], y=learning_module.predict(index_train).T, mode="lines", name="Predicted")
        return [x, y]
    
    #returns prediction trends
    def predict_trend(self):
        if self.data_object.get_all_prices().min() - self.data_object.get_all_prices().max() < 0:
            return "Price Predicted going Up"
        return "Price Predicted going Down"
