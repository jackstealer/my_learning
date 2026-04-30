import tkinter as tk

root = tk.Tk()
root.title("Motion Detector")
root.geometry("800x800")

def add():
    a = 10
    b = 20
    lbl.config(text=str(a + b))

def clear():
    lbl.config(text="")

# Button frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

btn = tk.Button(btn_frame, text="Click Me", command=add)
btn.pack(side="left", padx=5)

clear_btn = tk.Button(btn_frame, text="Clear", command=clear)
clear_btn.pack(side="left", padx=5)

lbl = tk.Label(root, text="KIET", font=(None, 18))
lbl.pack(pady=10)

root.mainloop()