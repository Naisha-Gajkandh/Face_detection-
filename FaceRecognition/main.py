import kivy
kivy.require('2.3.0')#write the version of kivy you have installed

# import all the kivy models required
from kivy.app import App # to create the app
from kivy.uix.boxlayout import BoxLayout # to create a box layout
from kivy.uix.gridlayout import GridLayout # to create a grid layout
from kivy.uix.label import Label # to create labels
from kivy.uix.textinput import TextInput # to take text input
from kivy.uix.button import Button # to create buttons
from kivy.uix.popup import Popup # to show messages
from kivy.clock import Clock # to schedule callbacks

import threading # to run functions in background threads
import os # to handle file paths

#import all the def func from other files 
import check_camera
import capture_image
import train_image
import recognize

print("Main.py started running") # to check if the file is running

def run_in_thread(fn, callback=None, *args, **kwargs): # a helper function (fn) to run a function in a background thread
   # Run 'fn' in background thread. When done, schedule callback(result) on main thread.
   # 'fn' should return (True/False, message).
    
    def _target():# '_' means this function is for internal use only not compulsory but a good practice
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            result = (False, str(e)) # if the function does not run properly it will return False and the error message
        if callback:
             Clock.schedule_once(lambda dt: callback(result), 0) 
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


class MainUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=10, padding=12, **kwargs)

        self.add_widget(Label(text="Face Recognition Attendance", font_size=22, size_hint=(1, 0.08)))

        grid = GridLayout(cols=2, spacing=10, size_hint=(1, 0.92))

        # Left: Buttons
        left = BoxLayout(orientation='vertical', spacing=8)
        btn_check = Button(text="Check Camera", size_hint=(1, None), height=48)
        btn_check.bind(on_release=lambda *_: run_in_thread(check_camera.camer, self.show_result))
        left.add_widget(btn_check)

        btn_capture = Button(text="Capture Faces (use Id & Name on right)", size_hint=(1, None), height=48)
        btn_capture.bind(on_release=lambda *_: self.on_capture())
        left.add_widget(btn_capture)

        btn_train = Button(text="Train Images", size_hint=(1, None), height=48)
        btn_train.bind(on_release=lambda *_: run_in_thread(train_image.TrainImages, self.show_result))
        left.add_widget(btn_train)

        btn_recognize = Button(text="Recognize & Attendance", size_hint=(1, None), height=48)
        btn_recognize.bind(on_release=lambda *_: run_in_thread(recognize.recognize_attendance, self.show_result))
        left.add_widget(btn_recognize)

        btn_quit = Button(text="Quit", size_hint=(1, None), height=48)
        btn_quit.bind(on_release=lambda *_: App.get_running_app().stop())
        left.add_widget(btn_quit)

        # Right: Inputs
        right = BoxLayout(orientation='vertical', spacing=8)
        right.add_widget(Label(text="Capture Inputs", size_hint=(1, None), height=24))
        self.input_id = TextInput(hint_text="Numeric ID (e.g. 1)", multiline=False, size_hint=(1, None), height=40)
        right.add_widget(self.input_id)
        self.input_name = TextInput(hint_text="Name (alphabetic)", multiline=False, size_hint=(1, None), height=40)
        right.add_widget(self.input_name)

        grid.add_widget(left)
        grid.add_widget(right)
        self.add_widget(grid)

    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', spacing=8)
        content.add_widget(Label(text=message))
        btn = Button(text='Close', size_hint=(1, 0.25))
        content.add_widget(btn)
        popup = Popup(title=title, content=content, size_hint=(0.7, 0.5))
        btn.bind(on_release=popup.dismiss)
        popup.open()

    def show_result(self, result):
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
            ok, msg = result
            title = "Success" if ok else "Error"
            self.show_popup(title, str(msg))
        else:
            self.show_popup("Result", str(result))

    def on_capture(self):
        Id = self.input_id.text.strip()
        name = self.input_name.text.strip()
        if not Id or not name:
            self.show_popup("Input required", "Please provide both ID and Name in the right panel.")
            return
        run_in_thread(lambda: capture_image.takeImages(Id=Id, name=name), self.show_result)

class FaceApp(App):
    def build(self):
        return MainUI()

if __name__ == "__main__":
    FaceApp().run()
