from bottle import route, run, abort
import datetime
from bottle import BaseRequest
from bottle import HTTPError
from bottle import post, request, response
from darkflow.net.build  import TFNet
from detector import detect_tiles
from bottle import HTTPResponse
import uuid
import os
from logging import basicConfig, getLogger
import logging
basicConfig(level=logging.DEBUG)
logger = getLogger(__name__)

BaseRequest.MEMFILE_MAX = 10240000

global count
UPLOAD_PATH = "./images/"
tfnet = None
@post('/upload')
def upload():
    print(request.files["upload"])
    upload = request.files.get('upload', '')
    if not upload:
        abort(400, "Upload file not found.")
    filename = upload.filename.lower()
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        abort(400, "File extensions must be '.png', '.jpg' or 'jpeg'.")
    
    path = get_upload_path(filename)
    upload.save(path)
    try:
        response = detect_tiles(tfnet, path)
    except HTTPError as e:
        os.remove(path)
        logger.debug("error")
        raise e
    # os.remove(path)
    logger.debug("ok")
    if not response[0]:
        return HTTPResponse(status=200, body={})
    return HTTPResponse(status=200, body=response[1])

def get_upload_path(filename):
    global count
    ext = "."+ filename.split(".")[-1]
    return UPLOAD_PATH + str(uuid.uuid4()) + ext
    # count += 1
    # return UPLOAD_PATH + str(count) + ext


if __name__ == "__main__":
    # count = 0
    tfnet_options = {"labels":"predictor/mahjong.names", "model":"predictor/yolov2.cfg", "load":"predictor/yolov2_5.weights", "threshold":0.5}
    tfnet = TFNet(tfnet_options)
    run(host="0.0.0.0", poort=8000)