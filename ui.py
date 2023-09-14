import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import mic
import final

class App:
    def __init__(self, master):
        self.master = master
        master.title("Select Module to Run")
        master.resizable(False, False)

        # Add icon to window
        img = Image.open("icon.png")
        img = img.resize((32, 32))
        icon = ImageTk.PhotoImage(img)
        master.iconphoto(False, icon)

        # Create main frame
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(padx=10, pady=10)

        # Add label to main frame
        self.label = ttk.Label(self.main_frame, text="Please select a module to run:")
        self.label.pack(pady=(0, 10))

        # Add mic button to main frame
        self.mic_img = Image.open("mic.png")
        self.mic_img = self.mic_img.resize((64, 64))
        self.mic_icon = ImageTk.PhotoImage(self.mic_img)
        self.mic_button = ttk.Button(self.main_frame, text="Run mic.py", image=self.mic_icon,
                                     compound=tk.TOP, command=self.run_mic)
        self.mic_button.pack(side=tk.LEFT, padx=(0, 20))

        # Add final button to main frame
        self.final_img = Image.open("final.png")
        self.final_img = self.final_img.resize((64, 64))
        self.final_icon = ImageTk.PhotoImage(self.final_img)
        self.final_button = ttk.Button(self.main_frame, text="Run final.py", image=self.final_icon,
                                       compound=tk.TOP, command=self.run_final)
        self.final_button.pack(side=tk.LEFT, padx=(0, 20))

        # Add quit button to main frame
        self.quit_button = ttk.Button(self.main_frame, text="Quit", command=master.quit)
        self.quit_button.pack(side=tk.LEFT)

        # Add status bar to bottom of window
        self.status = tk.StringVar()
        self.status.set("Ready")
        self.status_bar = ttk.Label(master, textvariable=self.status, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def run_mic(self):
        self.status.set("Running mic.py...")
        try:
            mic.main()
            self.status.set("mic.py finished")
        except Exception as e:
            self.status.set(f"Error running mic.py: {e}")

    def run_final(self):
        self.status.set("Running final.py...")
        try:
            final.main()
            self.status.set("final.py finished")
        except Exception as e:
            self.status.set(f"Error running final.py: {e}")

root = tk.Tk()
app = App(root)
root.mainloop()
