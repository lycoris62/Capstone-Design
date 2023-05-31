import os
from datetime import datetime
from glob import glob

import cv2
from flask import Flask, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename

from detection.object_detection import detect
from super_resolution.super_resolution import super_resolution

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 업로드 16MB 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CUR_DIR = os.path.abspath('.')
mainUrl = "/"


def is_error(request_files):
    if 'file' not in request_files:
        return "파일이 없습니다."

    f = request_files['file']
    if f.filename == '':
        return "파일이 없습니다."

    if not f.filename.split(".")[-1] in ALLOWED_EXTENSIONS:
        return "허용되지 않는 확장자입니다."
    return ""


def create_img_directory_name(now):
    return str(now).replace(" ", "_")


def create_img_id(now):
    return int(now.timestamp() * 1000000)


def parse_img_id(img_id):
    return datetime.fromtimestamp(img_id / 1000000)


def create_unique_directory_name(now):
    original_img_save_path = f'{CUR_DIR}/static/images/app/{create_img_directory_name(now)}'
    return {
        "original": original_img_save_path,
        "object": f'{original_img_save_path}/objects_original',
        "super": f'{original_img_save_path}/objects_super',
    }


def create_img_directory(now):
    path_names = create_unique_directory_name(now)
    os.makedirs(path_names["original"])
    os.makedirs(path_names["object"])
    os.makedirs(path_names["super"])
    return path_names


@app.route("/")
def hello():
    error_msg = request.args.get('error_msg', "")
    return render_template('index.html', error_msg=error_msg)


@app.route("/images")
def images():
    img_id = request.args.get("img_id", "")
    if img_id == "":
        return redirect(f"{mainUrl}?error_msg=잘못된 접근입니다.")

    directory_name = create_img_directory_name(parse_img_id(int(img_id)))
    ele_list = glob(f"static/images/app/{directory_name}/*")
    if len(ele_list) == 0:
        return redirect(f"{mainUrl}?error_msg=잘못된 접근입니다.")

    original_image = ""
    for ele in ele_list:
        if ele.split("/")[-1].startswith("original_"):
            original_image = ele[6:]
            break

    images = []
    obj_list = glob(f"static/images/app/{directory_name}/objects_original/*")
    for obj in obj_list:
        obj_filename = obj.split("/")[-1]
        images.append([
            f"images/app/{directory_name}/objects_original/{obj_filename}",
            f"images/app/{directory_name}/objects_super/{obj_filename}"
        ])

    return render_template(
        'images.html',
        original_image=original_image,
        images=images
    )


@app.route("/uploader", methods=["POST"])
def uploader_file():
    error_msg = is_error(request.files)
    if error_msg != "":
        return redirect(f"{mainUrl}?error_msg={error_msg}")

    f = request.files['file']
    now = datetime.now()
    paths = create_img_directory(now)

    original_save_to = f'{paths["original"]}/original_{secure_filename(f.filename)}'
    f.save(original_save_to)

    detect(cv2.imread(original_save_to), paths["object"], f.filename.split(".")[-1])
    super_resolution(paths["original"])

    return redirect(url_for('images', img_id=create_img_id(now)))


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0', port=4444)
