import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class App(tk.Canvas):
    def __init__(self, master=None, opt = None):
        self.pilImage = Image.open(list(opt["image_set"].values())[0])
        self.tkImage = ImageTk.PhotoImage(image=self.pilImage)

        super().__init__(master, width=self.pilImage.width, height=self.pilImage.height)
        self.root = master
        self.pack()

        self.create_image(0, 0, anchor="nw", image=self.tkImage)
        self.pack()
        self.bind("<Button-1>", self.callback)
        self.opt = opt
        self.win = None

    def callback(self, event):
        print("Pos: ", event.x, event.y)

        if(self.win != None):
            self.win.destroy()
        win = tk.Toplevel()
        self.win = win
        up_size = (opt["size"]*opt["scale"], opt["size"]*opt["scale"])

        for name, path in self.opt["image_set"].items():
            img = Image.open(path)
            img = img.crop((event.x, event.y, event.x+opt["size"], event.y+opt["size"]))
            img = img.resize(up_size)

            f = Figure((1, 1), dpi=200)  # 设置对比图像的显示大小(dpi)
            a = f.add_subplot(111)
            a.axis('off')
            f.suptitle(name, fontsize=6)
            a.imshow(img)

            canvas = FigureCanvasTkAgg(f, master=win)
            # canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        win.mainloop()


if __name__ == "__main__":
    opt = {
        # 设置需要对比的图像,请注意"HR"的名称不能改变,图像的分辨率大小应当一致,最好不要同时处理4张以上的图像
        "image_set": {
            "HR": "/home/tiger/Graduate/datasets/TestDataSR/Set5/baby.bmp",
            "SRCNN": "/home/tiger/Graduate/datasets/TestDataSR/Set5/baby.bmp",
            "Ours": "/home/tiger/Graduate/datasets/TestDataSR/Set5/baby.bmp"
        },
        "size": 50,  # 需要对比的图像块大小
        "scale": 2  # 放大倍数
    }
    root = tk.Tk()
    app = App(root, opt)
    root.mainloop()
