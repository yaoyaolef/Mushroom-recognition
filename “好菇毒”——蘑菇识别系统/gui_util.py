import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os


def create_root(img_path):
    if not os.path.isfile(img_path):
        print('no such file')
        raise AttributeError('no such image file')
    w = 1280
    h = 720
    root = tk.Tk()
    root.geometry('{}x{}'.format(w, h))
    root.wm_attributes('-fullscreen', 'true')
    canvas = tk.Canvas(root, width=w, height=h)
    canvas.pack()
    # photo = ImageTk.PhotoImage(Image.open(img_path).resize((w, h), Image.ANTIALIAS))
    photo = tk.PhotoImage(file=img_path)
    canvas.background = photo
    bg = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    return root  #, canvas


def show_video(cap, label, w, h, last_frame=None, interval=30, callback=None):
    lmain = label
    def show_frame():
        _, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        # img = Image.fromarray(cv2image)
        # imgtk = ImageTk.PhotoImage(image=img)
        # lmain.imgtk = imgtk
        # lmain.configure(image=imgtk)
        if last_frame is not None:
            last_frame[0] = frame
        show_image_on_label(frame, lmain, w, h)
        if callback is not None: callback()
        lmain.after(interval, show_frame)
    show_frame()


def show_image_on_label(img, label, w, h):
    if label is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)


def camera_cap():
    width, height = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
