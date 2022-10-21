import argparse

from utils import google_utils
from utils.datasets import *
from utils.utils import *
import tkinter as tk
from  tkinter  import ttk
from gui_util import *
import torch.backends.cudnn as cudnn
import time
import threading

class Mogu:
    def __init__(self) -> object:
        self.im1=cv2.imread('ui_img/1.png', cv2.IMREAD_COLOR)
        #self.V=True
        self.n=0
        self.p='x'
        self.result=cv2.imread('plants_img/3.png', cv2.IMREAD_COLOR)

    def detect(self,save_img=False):
        rotate_camera=90
        self.V=True
        global  xxx
        xxx=1
        xx=0
        path_result='plants_img/'
        result_img=['3.png','bds.png','cg.png','csg.png','hg.png','dyeg.png','htg.png','jyj.png','hhwegj.png','jzg.png','sbmg.png','pg.png','xbg.png','xg.png','ydj.png','hfmngj.png','zs.png']


        result_list=['']
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='weights/mogu1500.pt', help='model.pt path')
        parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        opt = parser.parse_args()
        out, source, weights, save_txt, imgsz = \
            opt.output, opt.source,opt.weights, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        # Initialize
        device = torch_utils.select_device(opt.device)
        #device = self.cc2
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        half=False
        # Load model
        google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32

        model.to(device).eval()
        if half:

            model.float()

        # Set Dataloader
        vid_path, vid_writer = None, None
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        ##########################################################
                        self.p=names[int(c)]
                        #############################################################
                    # Write results
                    for *xyxy, conf, cls in det:
                        if view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)



                # Stream results
                if view_img:

                    self.n+=1
                    self.im1=im0
                    xx=0
                    if self.p=='bds':
                        xx=1

                    elif self.p=='cg':
                        xx=2

                    elif self.p == 'csg':
                        xx=3
                    elif self.p=='hg':
                        xx=4

                    elif self.p == 'dyeg':
                        xx=5

                    elif self.p == 'htg':
                        xx = 6

                    elif self.p == 'jyj':
                        xx = 7

                    elif self.p=='hhwegj':
                        xx=8
                    elif self.p=='jzg':
                        xx=9

                    elif self.p == 'sbmg':
                        xx = 10

                    elif self.p=='pg':
                        xx=11
                    elif self.p=='xbg':
                        xx=12
                    elif self.p=='xg':
                        xx=13
                    elif self.p == 'ydj':
                        xx = 14

                    elif self.p=='hfmngj':
                        xx=15
                    elif self.p == 'zs':
                        xx = 16
                    else:
                        xx=0

                    p=path_result+result_img[xx]
                    self.result=cv2.imread(p, cv2.IMREAD_COLOR)
                            
       
    def main(self):

        self.root = create_root('./ui_img/beijing.png')
        photo0 = tk.PhotoImage(file='./ui_img/2.png')
        b2 = tk.Button(self.root, image=photo0, width=216, height=61, command=self.exit_click, borderwidth=0, bg='#6ABAFF')
        b2.place(x=1063,y=628)
        #b2.place(x=1350, y=5)
        


        #video_realtime↓
        self.video_label = tk.Label(self.root)

        #self.video_label.place(x=180, y=156, width=648, height=486)  #相机位置
        self.video_label.place(x=315, y=175, width=448, height=359)
        #video_realtime↑

        #result↓
        self.video_label_2 = tk.Label(self.root)
        #self.video_label_2.place(x=880, y=155, width=470, height=485)  #图片位置

        self.video_label_2.place(x=866, y=70, width=599, height=548)#yuan
        #result↑
        
        self.tick_callback()
        self.root.mainloop()

    def tick_callback(self):
        if self.n==0:
            self.n+=1
            #self.detect()
            t2 = threading.Thread(target=self.detect)
            t2.start()
            time.sleep(1)
        else:
            show_image_on_label(self.im1, self.video_label, 448,359)
            #show_image_on_label(self.result,self.video_label_2,470,485)  #图片大小
            show_image_on_label(self.result, self.video_label_2, 599, 548)  # yuan
            #show_image_on_label(self.result, self.video_label_2, 670, 485)
        self.root.after(50, self.tick_callback)

    def exit_click(self):
        global xxx
        xxx=2
        time.sleep(1)
        cv2.destroyAllWindows()
        self.root.destroy()
        

    def get_now(self):
        global  xxx      
        return xxx

    def start(self,cc1,cc2):
        self.cc1,self.cc2=cc1,cc2
        self.main()


if __name__ == '__main__':
   Mogu().main()


