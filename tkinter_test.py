from tkinter import *
import tkinter.ttk as ttk
import tkinter
from PIL import Image, ImageTk
import numpy as np


class Figure(object):
    FIGURES_BUNCH = {}

    def __init__(self, name, canvas):
        self.canvas = canvas
        self.name = name
        self.widget = -1
        self.FIGURES_BUNCH[name] = self
        self.anchor_point = np.array([0, 0])
        self.position = np.array([0, 0])
        self.angle = 0.0
        self.points = np.array([0, 0])

    def getPointsH(self):
        return np.hstack((self.points, np.ones((self.points.shape[0], 1))))

    def getID(self):
        return self.widget

    def setAnchorPoint(self, point=[0.0, 0.0]):
        self.anchor_point = np.array(point) - self.position

    def select(self, event):
        self.setAnchorPoint(np.array([event.x, event.y]))

    def unselect(self, event):
        self.setAnchorPoint()

    def move(self, position):
        self.position = position - self.anchor_point

    def rotate(self, angle, incremental=False):
        if not incremental:
            self.angle = angle
        else:
            self.angle += angle

    def update(self):
        pass

    @staticmethod
    def rotMatrix(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])

    @staticmethod
    def transformMatrix(position, angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), position[0]],
            [np.sin(angle), np.cos(angle), position[1]],
            [0, 0, 1.0]
        ])


class Anchor(Figure):
    def __init__(self, name, canvas, center, radius, color="#ff0000", parent_polygon=None):
        super(Anchor, self).__init__(name, canvas)

        p0 = np.array(center) - radius
        p1 = np.array(center) + radius
        self.points = np.array([
            p0,
            p1
        ]).reshape((-1, 2))

        if self.points.shape[0] <= 2:
            self.widget = self.canvas.create_oval(self.points[0, 0], self.points[0, 1], self.points[1, 0], self.points[1, 1],
                                                  outline=color, fill=color, width=2, stipple="gray50", tags=InteractiveCanvas.CANVAS_OBJECTS_TAG)
        else:
            self.widget = self.canvas.createPolygon(points, tags=InteractiveCanvas.CANVAS_OBJECTS_TAG)
        self.parent_polygon = parent_polygon

    def update(self):
        if self.parent_polygon is None:

            #trans = self.transformMatrix(self.position, self.angle)

            rotated_points = self.points  # np.matmul(self.getPointsH(), np.linalg.inv(trans))[:, :2]
            new_points = rotated_points + self.position
            new_points = tuple(new_points.ravel())
            self.canvas.coords(self.widget, new_points)
        else:
            new_points = self.points + self.position + self.parent_polygon.position
            new_points = np.matmul(new_points, Polygon.rotMatrix(self.angle + self.parent_polygon.angle))
            new_points = tuple(new_points.ravel())
            self.canvas.coords(self.widget, new_points)


class ReferenceFrame(Figure):

    def __init__(self, name, canvas, anchors=[]):
        super(ReferenceFrame, self).__init__(name, canvas)

        self.anchors = anchors
        self.widget = self.canvas.createPolygon([0, 0])

    def update(self):
        points = []

        if len(self.anchors) >= 3:

            p0_temp = self.anchors[0].position
            p1_temp = self.anchors[1].position
            p2_temp = self.anchors[2].position

            side_x = (p1_temp - p0_temp)
            dir_x = side_x / np.linalg.norm(side_x)

            dir_y = np.array([-dir_x[1], dir_x[0]])
            side_y_temp = p2_temp - p0_temp
            side_y = np.dot(dir_y, side_y_temp) * dir_y * 2

            p0 = p0_temp + side_y * 0.5
            p1 = p0 + side_x
            p2 = p1 - side_y
            p3 = p0 - side_y

            self.anchors[2].move((p0_temp + p1_temp) / 2.0)

            points.append(p0)
            points.append(p1)
            points.append(p2)
            points.append(p3)

            points = np.array(points)
            points = tuple(points.ravel())
            self.canvas.coords(self.widget, points)


class Polygon(object):

    def __init__(self, name, canvas, points, parent_polygon=None):
        self.name = name
        self.points = np.array(points).reshape((-1, 2))
        self.canvas = canvas

        if self.points.shape[0] <= 2:
            self.widget = self.canvas.create_oval(self.points[0, 0], self.points[0, 1], self.points[1, 0], self.points[1, 1],
                                                  fill="blue", outline="#DDD", width=4, tags=InteractiveCanvas.CANVAS_OBJECTS_TAG)
        else:
            self.widget = self.canvas.createPolygon(points, tags=InteractiveCanvas.CANVAS_OBJECTS_TAG)
        self.parent_polygon = parent_polygon

        self.position = np.array([0.0, 0.0])
        self.angle = 0.0

        #
        self.anchor_point = np.array([0, 0])

    def getID(self):
        return self.widget

    @staticmethod
    def rotMatrix(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])

    def select(self, event):
        self.setAnchorPoint(np.array([event.x, event.y]))

    def unselect(self, event):
        self.setAnchorPoint()

    def setAnchorPoint(self, point=[0.0, 0.0]):
        self.anchor_point = np.array(point) - self.position

    def rotate(self, angle, incremental=False):
        if not incremental:
            self.angle = angle
        else:
            self.angle += angle

    def move(self, position):
        self.position = position - self.anchor_point

    def transform(self, position=None, angle=None):
        if position is not None:
            self.position = position - self.anchor_point
        if angle is not None:
            self.angle = angle
        self.update()

    def update(self):
        if self.parent_polygon is None:
            rotated_points = np.matmul(self.points, Polygon.rotMatrix(self.angle))
            new_points = rotated_points + self.position
            new_points = tuple(new_points.ravel())
            self.canvas.coords(self.widget, new_points)
        else:
            new_points = self.points + self.position + self.parent_polygon.position
            new_points = np.matmul(new_points, Polygon.rotMatrix(self.angle + self.parent_polygon.angle))
            new_points = tuple(new_points.ravel())
            self.canvas.coords(self.widget, new_points)


class InteractiveCanvas(Canvas):
    CANVAS_OBJECTS_TAG = "CanvasObjects"

    def __init__(self, parent, sample_image, grow_factor=1.5):
        self.parent = parent
        self.sample_image = sample_image
        self.grow_factor = grow_factor
        self.image_width = self.sample_image.size[0]
        self.image_height = self.sample_image.size[1]
        self.width = self.image_width * grow_factor
        self.height = self.image_height * grow_factor
        self.picture = None
        self.picture_widget = None

        # Super Constructor
        super(InteractiveCanvas, self).__init__(parent, width=self.width, height=self.height)

        # Events
        self.bind('<Motion>', self.mouseMotion_)
        # self.tag_bind('canvas_object', '<B1-Motion>', self.dragMotion_)
        self.tag_bind(InteractiveCanvas.CANVAS_OBJECTS_TAG, "<ButtonPress-1>", lambda event: self.dragStart_(event, button=1))
        self.tag_bind(InteractiveCanvas.CANVAS_OBJECTS_TAG, "<ButtonPress-2>", lambda event: self.dragStart_(event, button=2))
        self.tag_bind(InteractiveCanvas.CANVAS_OBJECTS_TAG, "<ButtonPress-3>", lambda event: self.dragStart_(event, button=3))
        self.tag_bind(InteractiveCanvas.CANVAS_OBJECTS_TAG, "<B1-Motion>", self.dragMove_)
        self.tag_bind(InteractiveCanvas.CANVAS_OBJECTS_TAG, "<B2-Motion>", self.dragMove_)
        self.tag_bind(InteractiveCanvas.CANVAS_OBJECTS_TAG, "<B3-Motion>", self.dragMove_)
        self.tag_bind(InteractiveCanvas.CANVAS_OBJECTS_TAG, "<ButtonRelease-1>", self.dragStop_)

        # self.tag_bind("DnD", "<B1-Motion>", self.enter)

        # DRAG&DROP
        self.dragged_element = None

        # POLYGONS
        self.polygons = {}

    def debug(self):
        points = [
            -50, -50,
            50, -50,
            50, 50,
            -50, 50
        ]

        # pollo = Polygon("pollo", self, points)
        # self.polygons[pollo.getID()] = pollo

        points2 = [
            -30, -30,
            30, 30
        ]
        rf = ReferenceFrame("rf", self, [])
        a1 = Anchor("pollo1", self, [0.0, 0.0], 10, color="#ffffff")
        a2 = Anchor("pollo1", self, [0.0, 0.0], 10, color="#ff0000")
        a3 = Anchor("pollo1", self, [0.0, 0.0], 10, color="#00ff00")
        rf.anchors = [a1, a2, a3]
        self.addFigure(a1)
        self.addFigure(a2)
        self.addFigure(a3)
        self.addFigure(rf)

        a1.move([200, 200])
        a2.move([200, 100])
        a3.move([100, 200])
        self.updatePolygons()

    def addFigure(self, figure):
        self.polygons[figure.getID()] = figure

    def updatePolygons(self):
        for k, v in self.polygons.items():
            v.update()

    def canvasToImageConversion(self, point, output_type=int):
        return (np.array(point) / self.grow_factor).astype(output_type)

    def imageToCanvasConversion(self, point, output_type=int):
        return (np.array(point) * self.grow_factor).astype(output_type)

    def getCurrentPolygon(self):
        try:
            pid = self.find_withtag(tkinter.CURRENT)[0]
            return self.polygons[pid]
        except Exception as e:
            print(e)
            return None

    def mouseMotion_(self, evt):

        pass
        # p = [evt.x, evt.y]
        # print(p, self.canvasToImageConversion(p))

    def getCurrentObjectCoordinates(self):
        return np.array(self.coords(tkinter.CURRENT)).reshape((-1, 2))

    def getCurrentMousePosition(self, evt):
        return np.array([evt.x, evt.y])

    def mouseWheel_(self, evt):
        print(evt.__dict__)

    def dragStart_(self, evt, button):
        print(button)
        coords = self.getCurrentObjectCoordinates() - np.array([evt.x, evt.y])

        self.getCurrentPolygon().select(evt)

        self.dragged_element = {
            'widget': evt.widget,
            'coords': coords,
            'button': button,
            'start_position': self.getCurrentMousePosition(evt),
            'last_position': self.getCurrentMousePosition(evt)
        }

    def dragStop_(self, evt):
        self.dragged_element = None
        print("Drag stop", evt)

    def dragMove_(self, evt):
        if self.dragged_element is not None:
            #evt.widget.itemconfigure(tkinter.CURRENT, fill="blue")

            # # MOTION DISTANCE
            distance = self.getCurrentMousePosition(evt)[0] - self.dragged_element['start_position'][0]
            direction = self.getCurrentMousePosition(evt) - self.dragged_element['last_position']
            direction = direction / np.linalg.norm(direction)
            direction_x = direction[0]

            if self.dragged_element["button"] == 1:
                # MOVE
                self.getCurrentPolygon().move(self.getCurrentMousePosition(evt))
            elif self.dragged_element["button"] == 3:
                # ROTATE
                angle = (direction_x / 200) * np.pi
                pos = self.getCurrentPolygon().position
                self.getCurrentPolygon().rotate(angle, incremental=True)

            self.updatePolygons()

            # MOTION UPDATE
            self.dragged_element['last_position'] = self.getCurrentMousePosition(evt)
            # print(distance, angle)
            # rot = np.array([
            #     [np.cos(angle), -np.sin(angle)],
            #     [np.sin(angle), np.cos(angle)],
            # ])
            # coords = self.dragged_element['coords']
            # coords = np.matmul(coords, rot) + self.dragged_element['start_position']
            # coords = tuple(coords.astype(int).ravel())
            # evt.widget.coords(tkinter.CURRENT, coords)

            # coords = self.dragged_element['coords'] + np.array([evt.x, evt.y])
            # coords = tuple(coords.astype(int).ravel())
            # print(coords)
            # print(self.coords(tkinter.CURRENT))
            # print("#"*10)
            # evt.widget.coords(tkinter.CURRENT, coords)

            # print(self.coords(evt.widget))
            # evt.widget.coords(tkinter.CURRENT, 50, 50)
            # print(evt.x)
        print("Drag move", str(evt.widget))

    def showImage(self, image):
        resized_image = image.resize((int(self.width), int(self.height)))
        self.picture = ImageTk.PhotoImage(resized_image)
        self.picture_widget = self.create_image(self.width/2, self.height/2, image=self.picture)

    def createPolygon(self, points, outline='#f11', fill='#1f1', width=2, tags='', stipple="gray50"):
        return self.create_polygon(points, outline=outline, fill=fill, width=width, tags=InteractiveCanvas.CANVAS_OBJECTS_TAG, stipple=stipple)


class Application(Frame):
    def say_hi(self):
        print("hi there, everyone!")

    def drag_motion(self, data):
        print(data.__dict__)

    def mouse_move(self, data):
        # qprint(data.__dict__)
        # points = np.array([150, 100, 200, 120, 240, 180, 210,
        #                    200, 150, 150, 100, 200]).reshape((6, 2))
        # points = points + np.array([data.x, data.y])
        # points = list(points.ravel())
        # self.w.coords(self.obj1, *points)
        return
        print(data.__dict__)
        print(self.w.find_withtag(CURRENT))

    def enter(self, data):
        print("Enter", data.__dict__)

    def createWidgets(self):

        # im = Image.open("/home/daniele/Pictures/stand-alone-1280-720-3835.jpg")
        im = Image.open("/Users/daniele/Downloads/loop_dataset_2018/scan_01/images/0000001.jpg")

        grow = 1.5
        width, height = im.size

        self.w = InteractiveCanvas(self, sample_image=im)
        self.w.grid(row=0, sticky='sw')
        self.w['bg'] = "black"

        self.w.showImage(im)
        self.w.debug()
        # self.w.update()
        # w = self.w.winfo_width()
        # h = self.w.winfo_height()

        # ratio = float(im.size[0])/float(im.size[1])

        # nw = w
        # nh = nw / ratio
        # print("NRE", nw, nh)
        # im = im.resize((int(nw), int(nh)))

        # self.photo = ImageTk.PhotoImage(im)

        # self.imageFinal = self.w.create_image(w/2, h/2, image=self.photo)
        # #self.w.bind("<Motion>", self.mouse_move)

        # points = [150, 100, 300, 120, 240, 180, 210,
        #           200, 150, 150, 100, 200]
        # self.obj1 = self.w.create_polygon(points, outline='#f11', fill='#1f1', width=2, tags="DnD", stipple="gray50")
        # self.obj2 = self.w.create_oval(200, 200, 300, 300, tags="DnD")

        self.w.tag_bind("DnD", "<Button-1>", self.enter)
        # w.move(imageFinal, 20, 20)
        # w.update()
        # label = Label(master=w, image=photo)
        # label.image = photo  # keep a reference!
        # label.geometry('+%d+%d' % (20, 20))

        listbox = Listbox(self)
        listbox.grid(row=3, column=0)

        for a in range(10):
            listbox.insert(END, "item{}".format(a))
        listbox.bind("<B1-Motion>", self.drag_motion)

        self.ACC = ttk.Button(self)
        self.ACC["text"] = "ciao"
        self.ACC.grid(row=3, column=1)
        # self.ACC.pack({"side": "left"})

        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"] = "red"
        self.QUIT["bg"] = "#ffffee"
        self.QUIT["highlightthickness"] = 1
        self.QUIT["bd"] = 0
        self.QUIT["highlightbackground"] = "#ffffaa"

        self.QUIT["command"] = self.quit

        # self.QUIT.pack({"side": "left"})

        self.hi_there = Button(self)
        self.hi_there["text"] = "Hello",
        self.hi_there["command"] = self.say_hi

        # self.hi_there.pack({"side": "left"})

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=10)
        self.rowconfigure(1, weight=2)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()


root = Tk()
app = Application(master=root)

s = ttk.Style()
s.theme_use('clam')
print(s.theme_names())

app.mainloop()


root.destroy()
