from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
import pafy
import glob
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer



cfg = get_cfg()
cfg.merge_from_file('./output/config.yaml')
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
print(predictor)
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['kangaroowallaby','kangaroo','wallaby']
url = 'https://www.youtube.com/watch?v=fc-Lt6Hsgc0&t'
video = pafy.new(url)
best = video.getbest(preftype="mp4")
capture = cv2.VideoCapture(best.url)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('output.mp4', fourcc, 25, (640,480))
count = 0
imageCount = 0
while (True):
    grabbed, im = capture.read()
    if count == 30:
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('', v.get_image()[:, :, ::-1])
        out.write(v.get_image()[:, :, ::-1])
        count=0
        imageCount = imageCount + 1
        cv2.imwrite('./image/'+str(imageCount)+'.jpg', v.get_image()[:, :, ::-1])
    count = count + 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
out.release()
capture.release()
cv2.destroyAllWindows()
