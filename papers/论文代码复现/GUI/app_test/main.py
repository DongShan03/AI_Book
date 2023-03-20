from random import random
import kivy
import random
from kivy.app import App 
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout

kivy.require("1.9.0")

class MyRoot(BoxLayout):
    def __init__(self):
        super(MyRoot, self).__init__()
    
    def generate_number(self):
        self.random_label.text = str(random.randint(0, 1000))

class NeuralRandom(App):

    def build(self):
        return MyRoot()

neuralRandom = NeuralRandom()
neuralRandom.run()