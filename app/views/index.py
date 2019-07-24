import os
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from . import views


@views.route('/', methods=["GET"])
# @cache.cached(timeout=10)
def index():
    return "<h1>Hello Word!<h1>"


@views.route('/upload', methods=["GET", "post"])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 當前檔案所在路徑
        savepath = os.path.join(basepath, '../static/file_cache')
        if os.path.exists(savepath):
            os.makedirs(savepath)
        upload_path = os.path.join(basepath, '../static/file_cache',
                                   secure_filename(f.filename))  # 注意：沒有的資料夾一定要先建立，不然會提示沒有該路徑
        f.save(upload_path)
        return redirect(url_for('views.upload'))
    return render_template('upload.html')
