from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.graphics import Color, Rectangle
import random
from random import random as r
from functools import partial

class StressCanvasApp(App):
    seed = 0
    random.seed(0)
    def add_rects(self, label, wid, count, *args):
        label.text = str(int(label.text) + count)
        with wid.canvas:
            for x in range(count):
                Color(r(), 1.0, 1.0, mode="hsv")
                Rectangle(pos=(r()*wid.width + wid.x,
                                r()*wid.height + wid.y), size=(20, 20))
    
    def double_rects(self, label, wid, *args):
        count = int(label.text)
        self.add_rects(label, wid, count, *args)

    def reset_rects(self, label, wid, *args):
        label.text = "0"
        wid.canvas.clear()

    def change_seed(self, label, wid, *args):
        count = int(label.text)
        label.text = "0"
        self.seed += 1
        random.seed(self.seed)
        wid.canvas.clear()
        self.add_rects(label, wid, count, *args)

    
    def build(self):
        wid = Widget()
        label = Label(text="0")
        btn_add100 = Button(text="+ 100 rects", on_press=partial(self.add_rects, label, wid, 100))
        btn_add500 = Button(text="+ 500 rects", on_press=partial(self.add_rects, label, wid, 500))
        btn_double = Button(text="x 2", on_press=partial(self.double_rects, label, wid))
        btn_reset = Button(text="Reset", on_press=partial(self.reset_rects, label, wid))
        btn_seed = Button(text="Change seed", on_press=partial(self.change_seed, label, wid))

        layout = BoxLayout(size_hint=(1, None), height=50)
        layout.add_widget(btn_add100)
        layout.add_widget(btn_add500)
        layout.add_widget(btn_double)
        layout.add_widget(btn_reset)
        layout.add_widget(btn_seed)
        layout.add_widget(label)

        root = BoxLayout(orientation="vertical")
        root.add_widget(wid)
        root.add_widget(layout)
        return root

if __name__ == "__main__":
    StressCanvasApp().run()