from kivy.app import App
from kivy.lang import Builder

kv = '''
BoxLayout:
    orientation: 'vertical'
    BoxLayout:
        size_hint_y: None
        height: sp(100)
        BoxLayout: 
            orientation: 'vertical'
            Slider:
                id:e1
                min: -360.
                max: 360.
            Label:
                text: 'angle_start = {}'.format(round(e1.value, 1))
        BoxLayout:
            orientation:'vertical'
            Slider: 
                id: e2
                min: -360.
                max: 360.
                value: 360.
            Label: 
                text: 'angle_end = {}'.format(round(e2.value, 1))
    BoxLayout:
        size_hint_y: None
        height: sp(100)
        BoxLayout:
            orientation: 'vertical'
            Slider:
                id: wm
                min: 0
                max: 2
                value: 1
            Label:
                text: 'Width mult. = {}'.format(round(wm.value, 2))
        BoxLayout:
            orientation: 'vertical'
            Slider: 
                id: hm
                min: 0
                max: 2
                value: 1
            Label:
                text: 'Height mult. = {}'.format(round(hm.value, 2))
        Button:
            text: 'Reset ratios'
            on_press: wm.value = 1; hm.value = 1
    FloatLayout:
        canvas:
            Color:
                rgb: 1, 1, 1
            Ellipse:
                pos:20, 20
                size: 200 * wm.value, 201 * hm.value
                source: 'D:/pythonProject/AI_ebook/papers/论文代码复现/GUI/images/1.png'
                angle_start: e1.value
                angle_end: e2.value
'''

class CircleApp(App):
    def build(self):
        return Builder.load_string(kv)

CircleApp().run()