import json
import os
from datetime import datetime

import cv2
from flask import Flask, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename

from detection.object_detection import detect
from super_resolution.super_resolution import super_resolution

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 업로드 16MB 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CUR_DIR = os.path.abspath('.')

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/images")
def images():
    image_urls = request.args['image_urls']
    original_image = request.args['original_image'][1:-1]
    return render_template('images.html', original_image=original_image, images=json.loads(image_urls))

@app.route("/uploader", methods=["POST"])
def uploader_file():
    if 'file' not in request.files:
        return redirect(request.url)
    f = request.files['file']
    if f.filename == '':
        return redirect(request.url)
    if not f.filename.split(".")[-1] in ALLOWED_EXTENSIONS:
        return redirect(request.url)

    now_time = str(datetime.now()).replace(" ", "_")
    file_path = f'static/images/app/{now_time}'
    os.makedirs(file_path)
    save_to = f'{file_path}/{secure_filename(f.filename)}'
    image_save_path = os.path.join(CUR_DIR, save_to)
    f.save(image_save_path)
    original_image = cv2.imread(save_to)
    object_list = detect(original_image, file_path, f.filename.split(".")[-1])

    res = [f"images/app/{now_time}/objects_original/{object_img}" for object_img in object_list]
    super_resolution(res)
    res = [[ele, ele.replace("objects_original", "objects_super")] for ele in res]

    image_urls = json.dumps(res)
    origin_image = json.dumps("/".join(save_to.split("/")[1:]))

    return redirect(url_for('images', image_urls=image_urls, original_image=origin_image))

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0', port=4444)
